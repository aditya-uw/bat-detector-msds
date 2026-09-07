#!/usr/bin/env python3
"""Browser-based Folium deployment map framed by OSN AudioMoth spectrograms."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html
import io
import json
import math
import re
import tempfile
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

import folium
import fsspec
import numpy as np
import soundfile as sf
from folium.plugins import MeasureControl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


DEFAULT_CSV_PATH = Path("~/Downloads/STF Inventory - 08_19 Deployment Log-3.csv")
DEFAULT_OSN_PREFIX = "bio230143-bucket01/ubna_data_08/recover-20260824"
OSN_ENDPOINT = "https://sdsc.osn.xsede.org"
KING_COUNTY_TILES = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2025/MapServer/tile/{z}/{y}/{x}"
KING_COUNTY_ATTRIBUTION = "EagleView Technologies, Inc., King County"
NFFT_VALUES = (128, 256, 512, 1024, 2048, 4096, 8192)
PLOT_FONT_SIZE = 10
CACHE_DURATION_SECONDS = 300.0
DISPLAY_DURATION_SECONDS = 5.0
MAXIMUM_DISPLAY_DURATION_SECONDS = 60.0
CACHE_BLOCK_SECONDS = 5.0
EXAMPLE_FRAME_ORDER = {
    "top": ("008", "013", "023", "030", "028"),
    "left": ("011", "020", "024", "006"),
    "right": ("048", "033", "051", "039"),
    "bottom": ("046", "035", "036", "041", "042"),
}


@dataclass(frozen=True)
class Recorder:
    audiomoth: str
    sd_card: str
    post: str
    side: str
    latitude: float
    longitude: float
    corrected_bearing: float


@dataclass(frozen=True)
class Post:
    name: str
    latitude: float
    longitude: float
    corrected_bearing: float


@dataclass(frozen=True)
class AudioSegment:
    samples: np.ndarray
    sample_rate: int
    object_key: str


@dataclass(frozen=True)
class CachedRecording:
    path: Path
    sample_rate: int
    duration: float
    object_key: str


@dataclass(frozen=True)
class PanelRequest:
    panel_id: str
    recorder: Recorder


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def read_deployment_csv(csv_path: Path) -> tuple[dict[str, Recorder], dict[str, Post]]:
    with csv_path.expanduser().open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.reader(source))
    if len(rows) < 3 or len(rows[0]) < 37:
        raise ValueError("CSV does not have the expected two-row deployment-log header")
    header = [value.replace("\n", " ").strip().lower() for value in rows[0]]
    post_col = next(i for i, value in enumerate(header) if value.startswith("location") and "post" in value)
    side_col = next(i for i, value in enumerate(header) if "left/right" in value)
    corrected_col = next(i for i, value in enumerate(header) if "corrected" in value and "spyglass" in value)
    center_lat_col = next(i for i, value in enumerate(header) if "post center" in value and "latitude" in value and "garmin" in value)
    center_lon_col = next(i for i, value in enumerate(header) if "post center" in value and "longitude" in value and "garmin" in value)
    audiomoth_col = next(i for i, value in enumerate(header) if value == "audiomoth")
    sd_card_col = audiomoth_col + 1
    recorders = {}
    post_values = {}
    for row_number, row in enumerate(rows[2:], start=3):
        if len(row) <= sd_card_col or row[side_col].strip() not in {"L", "R"} or not row[post_col].strip():
            continue
        audiomoth = row[audiomoth_col].strip()
        sd_card = row[sd_card_col].strip()
        if not audiomoth.isdigit() or not sd_card.startswith("STF_"):
            continue
        try:
            latitude = float(row[center_lat_col])
            longitude = float(row[center_lon_col])
            corrected_bearing = float(row[corrected_col])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate or corrected bearing in CSV row {row_number}") from exc
        recorder = Recorder(audiomoth, sd_card, row[post_col].strip(), "left" if row[side_col].strip() == "L" else "right", latitude, longitude, corrected_bearing)
        if audiomoth in recorders:
            raise ValueError(f"Duplicate AudioMoth {audiomoth} in CSV")
        recorders[audiomoth] = recorder
        post_values.setdefault(recorder.post, set()).add((latitude, longitude, corrected_bearing))
    posts = {}
    for post_name, values in post_values.items():
        if len(values) != 1:
            raise ValueError(f"Post {post_name} has conflicting Garmin centers or corrected bearings")
        latitude, longitude, corrected_bearing = next(iter(values))
        posts[post_name] = Post(post_name, latitude, longitude, corrected_bearing)
    if len(recorders) != 50 or len(posts) != 25:
        raise ValueError(f"Expected 50 deployed AudioMoths at 25 posts; found {len(recorders)} at {len(posts)}")
    missing = set(sum((list(values) for values in EXAMPLE_FRAME_ORDER.values()), [])) - set(recorders)
    if missing:
        raise ValueError(f"Example layout AudioMoths missing from CSV: {sorted(missing)}")
    return recorders, posts


def destination_coordinate(latitude: float, longitude: float, bearing: float, length_metres: float) -> tuple[float, float]:
    angle = math.radians(bearing)
    destination_latitude = latitude + length_metres * math.cos(angle) / 111_320.0
    destination_longitude = longitude + length_metres * math.sin(angle) / (111_320.0 * math.cos(math.radians(latitude)))
    return destination_latitude, destination_longitude


def arrow_endpoint(post: Post, length_metres: float = 12.0) -> tuple[float, float]:
    return destination_coordinate(post.latitude, post.longitude, post.corrected_bearing, length_metres)


def parse_wav_datetime(object_key: str) -> dt.datetime:
    return dt.datetime.strptime(PurePosixPath(object_key).name, "%Y%m%d_%H%M%S.WAV")


def discover_audio_keys(filesys, osn_prefix: str, start_datetime: dt.datetime) -> dict[str, str]:
    keys = []
    hour = start_datetime.replace(minute=0, second=0, microsecond=0)
    for hours_back in range(3):
        filename = (hour - dt.timedelta(hours=hours_back)).strftime("%Y%m%d_%H%M%S.WAV")
        keys.extend(filesys.glob(f"{osn_prefix.rstrip('/')}/*/{filename}"))
    candidates = {}
    for key in keys:
        sd_card = PurePosixPath(key).parts[-2]
        file_datetime = parse_wav_datetime(key)
        if file_datetime <= start_datetime and (sd_card not in candidates or file_datetime > parse_wav_datetime(candidates[sd_card])):
            candidates[sd_card] = key
    return candidates


def read_audio_segment(filesys, object_key: str, start_datetime: dt.datetime, duration: float) -> AudioSegment:
    offset_seconds = (start_datetime - parse_wav_datetime(object_key)).total_seconds()
    with filesys.open(object_key, "rb") as remote_wav:
        with sf.SoundFile(remote_wav) as recorded_audio:
            sample_rate = recorded_audio.samplerate
            start_frame = round(offset_seconds * sample_rate)
            frame_count = round(duration * sample_rate)
            if start_frame < 0 or start_frame >= recorded_audio.frames:
                raise ValueError(f"requested time is outside {PurePosixPath(object_key).name}")
            recorded_audio.seek(start_frame)
            samples = recorded_audio.read(frame_count, dtype="float32")
    if len(samples) < frame_count:
        raise ValueError(f"only {len(samples) / sample_rate:.2f} seconds remain in {PurePosixPath(object_key).name}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return AudioSegment(samples, sample_rate, object_key)


def cache_audio_window(filesys, object_key: str, start_datetime: dt.datetime, duration: float, cache_path: Path) -> CachedRecording:
    first_object_key = object_key
    current_object_key = object_key
    offset_seconds = (start_datetime - parse_wav_datetime(object_key)).total_seconds()
    frames_remaining = None
    sample_rate = None
    cached_audio = None
    try:
        while frames_remaining is None or frames_remaining > 0:
            with filesys.open(current_object_key, "rb") as remote_wav:
                with sf.SoundFile(remote_wav) as recorded_audio:
                    if sample_rate is None:
                        sample_rate = recorded_audio.samplerate
                        frames_remaining = round(duration * sample_rate)
                        cached_audio = sf.SoundFile(cache_path, mode="w", samplerate=sample_rate, channels=1, format="WAV", subtype="PCM_16")
                    elif recorded_audio.samplerate != sample_rate:
                        raise ValueError(f"sample rate changed in {PurePosixPath(current_object_key).name}")
                    start_frame = round(offset_seconds * sample_rate)
                    if start_frame < 0 or start_frame >= recorded_audio.frames:
                        raise ValueError(f"requested time is outside {PurePosixPath(current_object_key).name}")
                    recorded_audio.seek(start_frame)
                    block_frames = round(CACHE_BLOCK_SECONDS * sample_rate)
                    while frames_remaining > 0:
                        samples = recorded_audio.read(min(block_frames, frames_remaining), dtype="float32")
                        if not len(samples):
                            break
                        if samples.ndim == 2:
                            samples = samples.mean(axis=1)
                        cached_audio.write(samples)
                        frames_remaining -= len(samples)
                    if frames_remaining > 0:
                        file_duration = recorded_audio.frames / sample_rate
                        next_datetime = parse_wav_datetime(current_object_key) + dt.timedelta(seconds=round(file_duration))
                        next_name = next_datetime.strftime("%Y%m%d_%H%M%S.WAV")
                        next_object_key = str(PurePosixPath(current_object_key).with_name(next_name))
                        if not filesys.exists(next_object_key):
                            cached_duration = duration - frames_remaining / sample_rate
                            raise ValueError(f"only {cached_duration:.2f} continuous seconds are available; missing {next_name}")
                        current_object_key = next_object_key
                        offset_seconds = 0.0
    finally:
        if cached_audio is not None:
            cached_audio.close()
    return CachedRecording(cache_path, sample_rate, duration, first_object_key)


def read_cached_segment(recording: CachedRecording, offset_seconds: float, duration: float = DISPLAY_DURATION_SECONDS) -> AudioSegment:
    with sf.SoundFile(recording.path) as cached_audio:
        start_frame = round(offset_seconds * cached_audio.samplerate)
        frame_count = round(duration * cached_audio.samplerate)
        if start_frame < 0 or start_frame + frame_count > cached_audio.frames:
            raise ValueError(f"display window {offset_seconds:.1f}–{offset_seconds + duration:.1f} seconds is outside the five-minute cache")
        cached_audio.seek(start_frame)
        samples = cached_audio.read(frame_count, dtype="float32")
    return AudioSegment(samples, recording.sample_rate, recording.object_key)


def parse_display_duration(parameters: dict[str, str]) -> float:
    duration = float(parameters.get("display_duration", str(DISPLAY_DURATION_SECONDS)))
    if duration < 0.1 or duration > MAXIMUM_DISPLAY_DURATION_SECONDS:
        raise ValueError(f"Display duration must be between 0.1 and {MAXIMUM_DISPLAY_DURATION_SECONDS:g} seconds")
    return duration


def example_layout(recorders: dict[str, Recorder]) -> dict[str, list[Recorder]]:
    return {edge: [recorders[audiomoth] for audiomoth in audiomoths] for edge, audiomoths in EXAMPLE_FRAME_ORDER.items()}


def all_post_layout(recorders: dict[str, Recorder], posts: dict[str, Post]) -> dict[str, list[Recorder]]:
    preferred = {recorders[audiomoth].post: audiomoth for audiomoths in EXAMPLE_FRAME_ORDER.values() for audiomoth in audiomoths}
    by_post = {}
    for recorder in recorders.values():
        by_post.setdefault(recorder.post, []).append(recorder)
    selected = []
    for post in sorted(posts, key=natural_key):
        candidates = sorted(by_post[post], key=lambda recorder: recorder.side)
        selected.append(recorders[preferred[post]] if post in preferred else candidates[0])
    return {"top": selected[:7], "right": selected[7:13], "bottom": list(reversed(selected[13:19])), "left": list(reversed(selected[19:]))}


def render_spectrogram(segment: AudioSegment, start_datetime: dt.datetime, utc_offset: float, nfft: int) -> str:
    figure = Figure(figsize=(3.6, 2.3), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    duration = len(segment.samples) / segment.sample_rate
    axis.specgram(segment.samples + 1e-6, Fs=segment.sample_rate, NFFT=nfft, noverlap=min(128, nfft // 2), cmap="jet", vmin=-90, vmax=-40, mode="magnitude", scale="dB")
    axis.set_ylim(0, min(48_000, segment.sample_rate / 2))
    axis.set_yticks((0, 24_000, 48_000))
    axis.set_yticklabels(("0", "24", "48"), fontsize=PLOT_FONT_SIZE)
    axis.set_ylabel("Frequency (kHz)", fontsize=PLOT_FONT_SIZE)
    ticks = np.linspace(0, duration, 4)
    display_start = start_datetime + dt.timedelta(hours=utc_offset)
    axis.set_xticks(ticks)
    tick_labels = axis.set_xticklabels([(display_start + dt.timedelta(seconds=float(value))).strftime("%M:%S") for value in ticks], fontsize=PLOT_FONT_SIZE)
    tick_labels[0].set_horizontalalignment("left")
    tick_labels[-1].set_horizontalalignment("right")
    axis.set_xlabel("Time (MM:SS)", fontsize=PLOT_FONT_SIZE)
    axis.tick_params(axis="both", pad=2, length=2)
    figure.subplots_adjust(left=0.18, right=0.96, top=0.97, bottom=0.23)
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=120, facecolor="white")
    return "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")


def build_folium_map(posts: dict[str, Post]) -> str:
    latitudes = [post.latitude for post in posts.values()]
    longitudes = [post.longitude for post in posts.values()]
    deployment_map = folium.Map(location=[sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)], zoom_start=18, control_scale=True, max_zoom=24, tiles=None)
    folium.TileLayer(tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png", attr="© OpenStreetMap contributors", name="OpenStreetMap", max_native_zoom=19, max_zoom=24, show=False).add_to(deployment_map)
    folium.TileLayer(tiles=KING_COUNTY_TILES, attr=KING_COUNTY_ATTRIBUTION, name="King County Aerial 2025", max_native_zoom=20, max_zoom=21, show=True).add_to(deployment_map)
    post_marker_names = {}
    for post in sorted(posts.values(), key=lambda value: natural_key(value.name)):
        endpoint = arrow_endpoint(post)
        left_wing = destination_coordinate(endpoint[0], endpoint[1], post.corrected_bearing + 150, 3.5)
        right_wing = destination_coordinate(endpoint[0], endpoint[1], post.corrected_bearing + 210, 3.5)
        shaft = [[post.latitude, post.longitude], list(endpoint)]
        arrowhead = [list(left_wing), list(endpoint), list(right_wing)]
        folium.PolyLine(locations=shaft, color="black", weight=5, opacity=0.9).add_to(deployment_map)
        folium.PolyLine(locations=arrowhead, color="black", weight=5, opacity=0.9).add_to(deployment_map)
        folium.PolyLine(locations=shaft, color="yellow", weight=3, opacity=1, tooltip=f"Corrected SpyGlass: {post.corrected_bearing:.1f}°").add_to(deployment_map)
        folium.PolyLine(locations=arrowhead, color="yellow", weight=3, opacity=1).add_to(deployment_map)
        post_marker = folium.CircleMarker(location=[post.latitude, post.longitude], radius=7, color="white", weight=1, fill=True, fill_color="#e53935", fill_opacity=1, tooltip=f"Post {post.name}: {post.corrected_bearing:.1f}° — drag to reassign its spectrogram")
        post_marker.add_to(deployment_map)
        post_marker_names[post.name] = post_marker.get_name()
    folium.LayerControl(position="topright").add_to(deployment_map)
    MeasureControl(position="topleft", primary_length_unit="meters", secondary_length_unit="feet").add_to(deployment_map)
    deployment_map.fit_bounds([[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]], padding=(25, 25))
    map_name = deployment_map.get_name()
    coordinates = {post.name: [post.latitude, post.longitude] for post in posts.values()}
    marker_variables = "{" + ",".join(f"{json.dumps(name)}:{variable}" for name, variable in post_marker_names.items()) + "}"
    connector_script = """
<script>
(function () {
    const deploymentMap = __MAP_NAME__;
    const postCoordinates = __COORDINATES__;
    const postMarkers = __POST_MARKERS__;
    let assignmentDrag = null;
    function sendPostPixels() {
        const points = {};
        Object.entries(postCoordinates).forEach(function (entry) {
            const point = deploymentMap.latLngToContainerPoint(entry[1]);
            points[entry[0]] = [point.x, point.y];
        });
        window.parent.postMessage({type: "folium-map-points", points: points}, window.location.origin);
    }
    function finishAssignmentDrag(event) {
        if (!assignmentDrag) return;
        let nearestPost = null;
        let nearestDistance = Infinity;
        const mousePoint = deploymentMap.latLngToContainerPoint(event.latlng);
        Object.entries(postCoordinates).forEach(function (entry) {
            const postPoint = deploymentMap.latLngToContainerPoint(entry[1]);
            const distance = mousePoint.distanceTo(postPoint);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestPost = entry[0];
            }
        });
        assignmentDrag.line.remove();
        deploymentMap.dragging.enable();
        deploymentMap.getContainer().style.cursor = "";
        if (nearestPost && nearestDistance <= 30 && nearestPost !== assignmentDrag.fromPost) {
            window.parent.postMessage({type: "reassign-post", fromPost: assignmentDrag.fromPost, toPost: nearestPost}, window.location.origin);
        }
        assignmentDrag = null;
    }
    Object.entries(postMarkers).forEach(function (entry) {
        entry[1].on("mousedown", function (event) {
            if (assignmentDrag) assignmentDrag.line.remove();
            L.DomEvent.stopPropagation(event.originalEvent);
            deploymentMap.dragging.disable();
            deploymentMap.getContainer().style.cursor = "crosshair";
            assignmentDrag = {fromPost: entry[0], line: L.polyline([event.latlng, event.latlng], {color: "yellow", weight: 4, dashArray: "8 5"}).addTo(deploymentMap)};
        });
    });
    deploymentMap.on("mousemove", function (event) {
        if (assignmentDrag) assignmentDrag.line.setLatLngs([postCoordinates[assignmentDrag.fromPost], event.latlng]);
    });
    deploymentMap.on("mouseup", finishAssignmentDrag);
    deploymentMap.on("move zoom resize moveend zoomend", sendPostPixels);
    window.addEventListener("load", function () { setTimeout(sendPostPixels, 300); });
    setTimeout(sendPostPixels, 600);
})();
</script>
""".replace("__MAP_NAME__", map_name).replace("__COORDINATES__", json.dumps(coordinates)).replace("__POST_MARKERS__", marker_variables)
    return deployment_map.get_root().render().replace("</html>", connector_script + "</html>")


class DeploymentWebApplication:
    def __init__(self, csv_path: Path, osn_prefix: str) -> None:
        self.csv_path = csv_path.expanduser()
        self.osn_prefix = osn_prefix
        self.recorders, self.posts = read_deployment_csv(self.csv_path)
        self.filesys = fsspec.filesystem("s3", anon=True, client_kwargs={"endpoint_url": OSN_ENDPOINT})
        self.map_html = build_folium_map(self.posts)
        self.cache_directory = tempfile.TemporaryDirectory(prefix="stf_spectrogram_cache_")
        self.recording_cache_key = None
        self.cached_recordings = {}
        self.cached_recording_pool = {}
        self.cached_errors = {}
        self.cached_pool_errors = {}
        self.audio_streams = {}
        self.audio_recording_pools = {}
        self.audio_bytes_cache = {}
        self.audio_lock = threading.Lock()
        preferred_audiomoths = {self.recorders[audiomoth].post: audiomoth for audiomoths in EXAMPLE_FRAME_ORDER.values() for audiomoth in audiomoths}
        self.preferred_recorder_by_post = {}
        for post_name in self.posts:
            candidates = sorted((recorder for recorder in self.recorders.values() if recorder.post == post_name), key=lambda recorder: recorder.side)
            self.preferred_recorder_by_post[post_name] = self.recorders[preferred_audiomoths[post_name]] if post_name in preferred_audiomoths else candidates[0]

    def create_audio_stream(self, recordings: dict[str, CachedRecording] | None = None, recording_pool: dict[str, CachedRecording] | None = None) -> str:
        stream_id = uuid.uuid4().hex
        with self.audio_lock:
            self.audio_streams[stream_id] = dict(recordings or {})
            self.audio_recording_pools[stream_id] = dict(recording_pool or {})
            while len(self.audio_streams) > 4:
                expired_stream = next(iter(self.audio_streams))
                del self.audio_streams[expired_stream]
                self.audio_recording_pools.pop(expired_stream, None)
                self.audio_bytes_cache = {key: value for key, value in self.audio_bytes_cache.items() if key[0] != expired_stream}
        return stream_id

    def register_cached_recording(self, stream_id: str, audiomoth: str, recording: CachedRecording) -> None:
        with self.audio_lock:
            if stream_id in self.audio_streams:
                self.audio_streams[stream_id][audiomoth] = recording
                self.audio_bytes_cache = {key: value for key, value in self.audio_bytes_cache.items() if key[:2] != (stream_id, audiomoth)}

    def register_pool_recording(self, stream_id: str, source_audiomoth: str, recording: CachedRecording) -> None:
        with self.audio_lock:
            if stream_id in self.audio_recording_pools:
                self.audio_recording_pools[stream_id][source_audiomoth] = recording

    def audio_wav(self, stream_id: str, audiomoth: str, offset_seconds: float, duration: float) -> bytes:
        if duration < 0.1 or duration > MAXIMUM_DISPLAY_DURATION_SECONDS:
            raise ValueError(f"Display duration must be between 0.1 and {MAXIMUM_DISPLAY_DURATION_SECONDS:g} seconds")
        cache_key = (stream_id, audiomoth, offset_seconds, duration)
        with self.audio_lock:
            if cache_key in self.audio_bytes_cache:
                return self.audio_bytes_cache[cache_key]
            recording = self.audio_streams.get(stream_id, {}).get(audiomoth)
        if recording is None:
            raise KeyError("Audio segment is no longer available; load the spectrograms again")
        segment = read_cached_segment(recording, offset_seconds, duration)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, segment.samples, segment.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_buffer.getvalue()
        with self.audio_lock:
            self.audio_bytes_cache[cache_key] = wav_bytes
        return wav_bytes

    def replace_recording_cache(self, cache_key, recordings: dict[str, CachedRecording], errors: dict[str, str], recording_pool: dict[str, CachedRecording], pool_errors: dict[str, str]) -> None:
        old_paths = {recording.path for recording in self.cached_recording_pool.values()}
        new_paths = {recording.path for recording in recording_pool.values()}
        with self.audio_lock:
            active_paths = {recording.path for stream_pool in self.audio_recording_pools.values() for recording in stream_pool.values()}
        for old_path in old_paths - new_paths - active_paths:
            old_path.unlink(missing_ok=True)
        self.recording_cache_key = cache_key
        self.cached_recordings = recordings
        self.cached_recording_pool = recording_pool
        self.cached_errors = errors
        self.cached_pool_errors = pool_errors

    def layout_for(self, mode: str) -> dict[str, list[Recorder]]:
        return example_layout(self.recorders) if mode == "example" else all_post_layout(self.recorders, self.posts)

    def panel_requests(self, parameters: dict[str, str]) -> list[PanelRequest]:
        try:
            assignments = json.loads(parameters.get("assignments", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("Post assignments are invalid") from exc
        if not isinstance(assignments, dict):
            raise ValueError("Post assignments must be a dictionary")
        layout = self.layout_for(parameters["layout"])
        requests = []
        for edge in ("top", "left", "right", "bottom"):
            for original_recorder in layout[edge]:
                target_post = assignments.get(original_recorder.audiomoth, original_recorder.post)
                if target_post not in self.preferred_recorder_by_post:
                    raise ValueError(f"Unknown assigned post {target_post}")
                requests.append(PanelRequest(original_recorder.audiomoth, self.preferred_recorder_by_post[target_post]))
        return requests

    @staticmethod
    def panel_title(recorder: Recorder) -> str:
        return f"Audiomoth {recorder.audiomoth}, SD card {recorder.sd_card} at site {recorder.post} ({recorder.side})"

    def stream_plot(self, parameters: dict[str, str], send) -> None:
        start_datetime = dt.datetime.strptime(parameters["start"], "%Y-%m-%d %H:%M:%S")
        nfft = int(parameters["nfft"])
        utc_offset = float(parameters["utc_offset"])
        display_duration = parse_display_duration(parameters)
        if nfft not in NFFT_VALUES:
            raise ValueError(f"NFFT must be one of {NFFT_VALUES}")
        requests = self.panel_requests(parameters)
        total = len(requests)
        cache_key = (parameters["start"], parameters["osn_prefix"])

        if cache_key == self.recording_cache_key:
            recording_pool = dict(self.cached_recording_pool)
            pool_errors = dict(self.cached_pool_errors)
            recordings = {request.panel_id: recording_pool[request.recorder.audiomoth] for request in requests if request.recorder.audiomoth in recording_pool}
            errors = {request.panel_id: pool_errors[request.recorder.audiomoth] for request in requests if request.recorder.audiomoth in pool_errors}
            stream_id = self.create_audio_stream(recordings, recording_pool)
            print(f"[load] Reusing five-minute caches for {len(recording_pool)}/25 posts.", flush=True)
            send({"type": "session", "stream_id": stream_id, "offset": 0.0})
            send({"type": "status", "text": f"Using cached five-minute audio; plotting {total} spectrograms..."})
            for index, request in enumerate(requests, start=1):
                if request.panel_id in recordings:
                    self._send_panel(send, stream_id, request, recordings[request.panel_id], start_datetime, utc_offset, nfft, 0.0, display_duration)
                    print(f"[plot {index}/{total}] Displayed seconds 0–{display_duration:g} for panel {request.panel_id} from {request.recorder.post}.", flush=True)
                else:
                    send({"type": "panel", "panel_id": request.panel_id, "title": self.panel_title(request.recorder), "error": errors.get(request.panel_id, "Recording unavailable")})
            status = f"Showing seconds 0–{display_duration:g} from {len(recordings)}/{total} five-minute panel caches" + (f"; {len(errors)} unavailable" if errors else "")
            send({"type": "done", "text": status, "stream_id": stream_id, "offset": 0.0})
            print(f"[plot] Finished drawing {total} recording panels.", flush=True)
            return

        stream_id = self.create_audio_stream()
        send({"type": "session", "stream_id": stream_id, "offset": 0.0})
        unique_recorders = {recorder.audiomoth: recorder for recorder in self.preferred_recorder_by_post.values()}
        panel_ids_by_audiomoth = {}
        request_by_panel = {request.panel_id: request for request in requests}
        for request in requests:
            panel_ids_by_audiomoth.setdefault(request.recorder.audiomoth, []).append(request.panel_id)
        print(f"[load] Searching OSN for five minutes from all {len(unique_recorders)} post recordings at {start_datetime}...", flush=True)
        send({"type": "status", "text": f"Searching OSN and caching five minutes from all {len(unique_recorders)} posts..."})
        keys = discover_audio_keys(self.filesys, parameters["osn_prefix"], start_datetime)
        print(f"[load] Found candidate WAV files for {len(keys)} SD cards.", flush=True)
        recordings = {}
        recording_pool = {}
        errors = {}
        pool_errors = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for audiomoth, recorder in unique_recorders.items():
                panel_ids = panel_ids_by_audiomoth.get(audiomoth, [])
                if recorder.sd_card not in keys:
                    error = f"No WAV found for {recorder.sd_card} near {start_datetime}"
                    pool_errors[audiomoth] = error
                    for panel_id in panel_ids:
                        errors[panel_id] = error
                        send({"type": "panel", "panel_id": panel_id, "title": self.panel_title(recorder), "error": error})
                    print(f"[load] AudioMoth {recorder.audiomoth}, {recorder.sd_card}: no WAV found", flush=True)
                else:
                    cache_path = Path(self.cache_directory.name) / f"{stream_id}_{recorder.audiomoth}.wav"
                    future = executor.submit(cache_audio_window, self.filesys, keys[recorder.sd_card], start_datetime, CACHE_DURATION_SECONDS, cache_path)
                    futures[future] = (recorder, panel_ids, cache_path)
            for index, future in enumerate(as_completed(futures), start=1):
                recorder, panel_ids, cache_path = futures[future]
                try:
                    recording = future.result()
                    print(f"[load {index}/{len(futures)}] Cached 5 minutes: AudioMoth {recorder.audiomoth}, {recorder.sd_card}, {recorder.post} ({recorder.side})", flush=True)
                    recording_pool[recorder.audiomoth] = recording
                    self.register_pool_recording(stream_id, recorder.audiomoth, recording)
                    for panel_id in panel_ids:
                        request = request_by_panel[panel_id]
                        recordings[panel_id] = recording
                        self.register_cached_recording(stream_id, panel_id, recording)
                        self._send_panel(send, stream_id, request, recording, start_datetime, utc_offset, nfft, 0.0, display_duration)
                        print(f"[plot] Displayed seconds 0–{display_duration:g} in panel {panel_id} from {recorder.post}.", flush=True)
                except Exception as exc:
                    cache_path.unlink(missing_ok=True)
                    pool_errors[recorder.audiomoth] = str(exc)
                    for panel_id in panel_ids:
                        errors[panel_id] = str(exc)
                        send({"type": "panel", "panel_id": panel_id, "title": self.panel_title(recorder), "error": str(exc)})
                    print(f"[load {index}/{len(futures)}] ERROR AudioMoth {recorder.audiomoth}: {exc}", flush=True)
                send({"type": "status", "text": f"Cached {index}/{len(futures)} post recordings; displayed {len(recordings)}/{total} panels..."})
        self.replace_recording_cache(cache_key, recordings, errors, recording_pool, pool_errors)
        status = f"Cached {len(recording_pool)}/25 posts; showing seconds 0–{display_duration:g} in {len(recordings)}/{total} panels" + (f"; {len(pool_errors)} unavailable" if pool_errors else "")
        send({"type": "done", "text": status, "stream_id": stream_id, "offset": 0.0})
        print(f"[plot] Finished drawing {total} recording panels.", flush=True)

    def _send_panel(self, send, stream_id: str, request: PanelRequest, recording: CachedRecording, start_datetime: dt.datetime, utc_offset: float, nfft: int, offset_seconds: float, display_duration: float) -> None:
        segment = read_cached_segment(recording, offset_seconds, display_duration)
        display_datetime = start_datetime + dt.timedelta(seconds=offset_seconds)
        image = render_spectrogram(segment, display_datetime, utc_offset, nfft)
        audio_url = f"/audio?stream={stream_id}&audiomoth={request.panel_id}&offset={offset_seconds:g}&duration={display_duration:g}"
        send({"type": "panel", "panel_id": request.panel_id, "title": self.panel_title(request.recorder), "post": request.recorder.post, "image": image, "audio_url": audio_url})

    def stream_scroll(self, parameters: dict[str, str], send) -> None:
        stream_id = parameters["stream"]
        offset_seconds = float(parameters["offset"])
        nfft = int(parameters["nfft"])
        utc_offset = float(parameters["utc_offset"])
        display_duration = parse_display_duration(parameters)
        start_datetime = dt.datetime.strptime(parameters["start"], "%Y-%m-%d %H:%M:%S")
        maximum_offset = CACHE_DURATION_SECONDS - display_duration
        if nfft not in NFFT_VALUES:
            raise ValueError(f"NFFT must be one of {NFFT_VALUES}")
        if offset_seconds < 0 or offset_seconds > maximum_offset:
            raise ValueError(f"Offset must be between 0 and {maximum_offset:g} seconds")
        with self.audio_lock:
            recordings = dict(self.audio_streams.get(stream_id, {}))
        if not recordings:
            raise ValueError("The five-minute audio cache expired; load the recordings again")
        requests = self.panel_requests(parameters)
        send({"type": "session", "stream_id": stream_id, "offset": offset_seconds})
        send({"type": "status", "text": f"Moving spectrograms to seconds {offset_seconds:g}–{offset_seconds + display_duration:g}..."})
        print(f"[scroll] Requested seconds {offset_seconds:g}–{offset_seconds + display_duration:g} for {len(requests)} recording panels.", flush=True)
        plotted = 0
        failed = 0
        for request in requests:
            if request.panel_id not in recordings:
                continue
            try:
                self._send_panel(send, stream_id, request, recordings[request.panel_id], start_datetime, utc_offset, nfft, offset_seconds, display_duration)
            except (FileNotFoundError, RuntimeError, sf.LibsndfileError) as exc:
                failed += 1
                send({"type": "panel", "panel_id": request.panel_id, "title": self.panel_title(request.recorder), "error": f"Cached recording unavailable: {exc}"})
                print(f"[scroll] ERROR panel {request.panel_id} from {request.recorder.post}: {exc}", flush=True)
                continue
            plotted += 1
            print(f"[scroll {plotted}/{len(recordings)}] Displayed seconds {offset_seconds:g}–{offset_seconds + display_duration:g}: panel {request.panel_id} from {request.recorder.post}.", flush=True)
        status = f"Showing seconds {offset_seconds:g}–{offset_seconds + display_duration:g} from {plotted} recordings" + (f"; {failed} cache errors" if failed else "")
        send({"type": "done", "text": status, "stream_id": stream_id, "offset": offset_seconds})
        print(f"[scroll] Finished updating {plotted} recording panels.", flush=True)

    def stream_reassignment(self, parameters: dict[str, str], send) -> None:
        panel_id = parameters["panel_id"]
        target_post = parameters["to_post"]
        start_datetime = dt.datetime.strptime(parameters["start"], "%Y-%m-%d %H:%M:%S")
        offset_seconds = float(parameters.get("offset", "0"))
        nfft = int(parameters["nfft"])
        utc_offset = float(parameters["utc_offset"])
        display_duration = parse_display_duration(parameters)
        if target_post not in self.preferred_recorder_by_post:
            raise ValueError(f"Unknown assigned post {target_post}")
        if offset_seconds < 0 or offset_seconds > CACHE_DURATION_SECONDS - display_duration:
            raise ValueError("Current display offset is outside the five-minute cache")
        recorder = self.preferred_recorder_by_post[target_post]
        request = PanelRequest(panel_id, recorder)
        stream_id = parameters.get("stream", "")
        with self.audio_lock:
            session_recordings = dict(self.audio_streams.get(stream_id, {}))
            recording_pool = dict(self.audio_recording_pools.get(stream_id, {}))
        if stream_id not in self.audio_streams:
            raise ValueError("The five-minute audio cache expired; load the recordings again")
        send({"type": "session", "stream_id": stream_id, "offset": offset_seconds})
        send({"type": "status", "text": f"Switching panel {panel_id} to cached post {target_post}..."})

        recording = recording_pool.get(recorder.audiomoth)
        if recording is None or not recording.path.exists():
            raise ValueError(f"Post {target_post} was unavailable during the initial 25-post load; load the recordings again to retry it")
        print(f"[reassign] Using preloaded AudioMoth {recorder.audiomoth}, {recorder.sd_card} for panel {panel_id}.", flush=True)

        session_recordings[panel_id] = recording
        self.register_cached_recording(stream_id, panel_id, recording)
        self._send_panel(send, stream_id, request, recording, start_datetime, utc_offset, nfft, offset_seconds, display_duration)
        updated_errors = {key: value for key, value in self.cached_errors.items() if key != panel_id}
        self.replace_recording_cache((parameters["start"], parameters["osn_prefix"]), session_recordings, updated_errors, recording_pool, self.cached_pool_errors)
        status = f"Panel {panel_id} now shows post {target_post}, seconds {offset_seconds:g}–{offset_seconds + display_duration:g}"
        send({"type": "done", "text": status, "stream_id": stream_id, "offset": offset_seconds})
        print(f"[reassign] Displayed post {target_post} in panel {panel_id}.", flush=True)

    def render_page(self, parameters: dict[str, str], images: dict[str, str] | None = None, errors: dict[str, str] | None = None, status: str = "Ready") -> str:
        images = images or {}
        errors = errors or {}
        layout = self.layout_for(parameters["layout"])
        requests_by_panel = {request.panel_id: request for request in self.panel_requests(parameters)}
        edge_html = {edge: "".join(self._panel_html(recorder, requests_by_panel[recorder.audiomoth].recorder, edge, images.get(recorder.audiomoth), errors.get(recorder.audiomoth)) for recorder in layout[edge]) for edge in ("top", "left", "right", "bottom")}
        options = "".join(f'<option value="{value}"{" selected" if str(value) == parameters["nfft"] else ""}>{value}</option>' for value in NFFT_VALUES)
        replacements = {
            "__START__": html.escape(parameters["start"], quote=True),
            "__SCROLL_STEP__": html.escape(parameters["scroll_step"], quote=True),
            "__DISPLAY_DURATION__": html.escape(parameters["display_duration"], quote=True),
            "__ASSIGNMENTS__": html.escape(parameters.get("assignments", "{}"), quote=True),
            "__UTC_OFFSET__": html.escape(parameters["utc_offset"], quote=True),
            "__OSN_PREFIX__": html.escape(parameters["osn_prefix"], quote=True),
            "__NFFT_OPTIONS__": options,
            "__EXAMPLE_SELECTED__": " selected" if parameters["layout"] == "example" else "",
            "__ALL_SELECTED__": " selected" if parameters["layout"] == "all" else "",
            "__STATUS__": html.escape(status),
            "__TOP__": edge_html["top"],
            "__LEFT__": edge_html["left"],
            "__RIGHT__": edge_html["right"],
            "__BOTTOM__": edge_html["bottom"],
        }
        page = PAGE_TEMPLATE
        for placeholder, value in replacements.items():
            page = page.replace(placeholder, value)
        return page

    @staticmethod
    def _panel_html(original_recorder: Recorder, assigned_recorder: Recorder, edge: str, image: str | None, error: str | None) -> str:
        title = DeploymentWebApplication.panel_title(assigned_recorder)
        body = f'<img src="{image}" alt="Spectrogram for AudioMoth {assigned_recorder.audiomoth}">' if image else f'<div class="placeholder">{html.escape(error or "Click Load and plot")}</div>'
        return f'<article id="panel-{original_recorder.audiomoth}" class="spectrogram" data-panel-id="{original_recorder.audiomoth}" data-post="{html.escape(assigned_recorder.post, quote=True)}" data-edge="{edge}"><div class="panel-title"><span>{html.escape(title)}</span><button type="button" class="audio-button" onclick="toggleAudio(this)" disabled aria-label="Play AudioMoth {assigned_recorder.audiomoth}">▶</button></div><div class="plot-body">{body}</div></article>'


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STF Deployment Map and Spectrogram Viewer</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; background: #eef1f4; color: #202124; font-family: Arial, sans-serif; }
.controls { position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap; align-items: end; gap: 8px 14px; padding: 10px 16px; background: rgba(255,255,255,.97); border-bottom: 1px solid #bbb; box-shadow: 0 1px 5px rgba(0,0,0,.18); }
.control { display: flex; flex-direction: column; gap: 3px; font-size: 12px; font-weight: bold; }
.control input, .control select, button { height: 30px; padding: 4px 7px; font-size: 13px; }
.prefix { flex: 1 1 410px; }.prefix input { width: 100%; }
button { padding: 4px 16px; font-weight: bold; cursor: pointer; }
#status { flex-basis: 100%; min-height: 16px; font-size: 13px; }
.frame { position: relative; display: grid; grid-template-columns: var(--panel-width,190px) minmax(480px,1fr) var(--panel-width,190px); grid-template-areas: "top top top" "left map right" "bottom bottom bottom"; gap: 18px 14px; min-width: 1050px; padding: 18px 32px 36px; background: #202428; }
.edge { position: relative; z-index: 4; }
.top { grid-area: top; display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
.bottom { grid-area: bottom; display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
.left { grid-area: left; display: flex; flex-direction: column; align-items: flex-start; justify-content: space-between; gap: 28px; }
.right { grid-area: right; display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between; gap: 28px; }
.map-wrap { grid-area: map; position: relative; z-index: 2; min-height: 720px; border: 1px solid #777; background: #cfd8dc; box-shadow: 0 1px 4px rgba(0,0,0,.25); }
.map-wrap iframe { display: block; width: 100%; height: 100%; min-height: 720px; border: 0; }
.spectrogram { flex: 0 0 auto; width: var(--panel-width, 190px); aspect-ratio: 1.5; display: flex; flex-direction: column; overflow: hidden; background: white; border: 1px solid #c4c7ca; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.panel-title { position: relative; flex: 0 0 auto; min-height: 26px; display: flex; align-items: center; justify-content: center; padding: 5px 27px 2px; font-size: 11px; font-weight: bold; line-height: 1.1; text-align: center; }
.audio-button { position: absolute; top: 2px; right: 3px; width: 22px; height: 22px; padding: 0; border: 1px solid #777; border-radius: 50%; background: #f4f4f4; color: #202124; font-size: 11px; line-height: 20px; cursor: pointer; }
.audio-button:hover:not(:disabled) { background: #dce8f8; }
.audio-button:disabled { color: #aaa; cursor: default; }
.audio-button.playing { background: #d32f2f; border-color: #8b1111; color: white; }
.plot-body { min-height: 0; flex: 1 1 auto; display: flex; }
.plot-body img { display: block; width: 100%; height: 100%; object-fit: fill; }
.placeholder { width: 100%; display: flex; align-items: center; justify-content: center; padding: 10px; color: #666; font-size: 12px; text-align: center; background: #fafafa; }
#connections { position: absolute; inset: 0; z-index: 3; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
</style>
</head>
<body>
<form id="plot-form" class="controls" action="/" method="get" onsubmit="startLoading(event); return false;">
<input id="assignments" type="hidden" name="assignments" value="__ASSIGNMENTS__">
<label class="control"><span>Start (WAV/UTC)</span><input name="start" value="__START__" size="20" oninput="invalidateCurrentCache()"></label>
<label class="control"><span>Step (s)</span><input id="scroll-step" name="scroll_step" value="__SCROLL_STEP__" size="7"></label>
<label class="control"><span>Duration (s)</span><input id="display-duration" name="display_duration" type="number" min="0.1" max="60" step="0.1" value="__DISPLAY_DURATION__" size="7" onchange="refreshCurrentSpectrograms()"></label>
<label class="control"><span>NFFT</span><select name="nfft">__NFFT_OPTIONS__</select></label>
<label class="control"><span>Display UTC offset</span><input name="utc_offset" value="__UTC_OFFSET__" size="5"></label>
<label class="control"><span>Layout</span><select name="layout" onchange="reloadLayout()"><option value="example"__EXAMPLE_SELECTED__>Example order (18)</option><option value="all"__ALL_SELECTED__>All posts (25)</option></select></label>
<label class="control prefix"><span>OSN prefix</span><input name="osn_prefix" value="__OSN_PREFIX__" oninput="invalidateCurrentCache()"></label>
<button id="plot-button" type="submit">Load 5 minutes</button>
<button id="backward-button" type="button" onclick="stepSpectrograms(-1)" disabled>← Backward</button>
<button id="forward-button" type="button" onclick="stepSpectrograms(1)" disabled>Forward →</button>
<div id="status">__STATUS__</div>
</form>
<main class="frame" id="frame">
<section class="edge top">__TOP__</section>
<section class="edge left">__LEFT__</section>
<div class="map-wrap"><iframe id="map-frame" src="/map" title="Interactive deployment map"></iframe></div>
<section class="edge right">__RIGHT__</section>
<section class="edge bottom">__BOTTOM__</section>
<svg id="connections" aria-hidden="true"></svg>
</main>
<script>
let latestPoints = null;
let activeAudio = null;
let activeAudioButton = null;
let currentAudioStream = null;
let currentOffset = 0;
let activePlotStream = null;
let plotOperationNumber = 0;
const CACHE_DURATION_SECONDS = 300;
function selectedDisplayDuration() {
    return Number(document.getElementById("display-duration").value);
}
function maximumOffset() {
    return CACHE_DURATION_SECONDS - selectedDisplayDuration();
}
function updateNavigationButtons() {
    document.getElementById("backward-button").disabled = !currentAudioStream || currentOffset <= 0;
    document.getElementById("forward-button").disabled = !currentAudioStream || currentOffset >= maximumOffset();
}
function equalizePanelSizes() {
    const frame = document.getElementById("frame");
    const top = document.querySelector(".top");
    const styles = window.getComputedStyle(frame);
    const contentWidth = frame.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
    const panelCount = top.querySelectorAll(".spectrogram").length;
    const gap = parseFloat(window.getComputedStyle(top).gap);
    const panelWidth = (contentWidth - gap * (panelCount - 1)) / panelCount;
    frame.style.setProperty("--panel-width", panelWidth + "px");
}
function reloadLayout() {
    const parameters = new URLSearchParams(new FormData(document.getElementById("plot-form")));
    window.location.href = "/?" + parameters.toString();
}
function invalidateCurrentCache() {
    currentAudioStream = null;
    currentOffset = 0;
    document.getElementById("backward-button").disabled = true;
    document.getElementById("forward-button").disabled = true;
    document.getElementById("status").textContent = "Start time changed; click Load 5 minutes to create a new cache.";
}
function reassignPost(fromPost, toPost) {
    if (document.getElementById("plot-button").disabled) {
        document.getElementById("status").textContent = "Wait for the current plot update to finish before reassigning a post.";
        return;
    }
    const panels = Array.from(document.querySelectorAll(".spectrogram")).filter(function (panel) { return panel.dataset.post === fromPost; });
    if (!panels.length) {
        document.getElementById("status").textContent = "No displayed spectrogram is attached to post " + fromPost + ".";
        return;
    }
    const panel = panels[0];
    const assignmentsInput = document.getElementById("assignments");
    let assignments = {};
    try { assignments = JSON.parse(assignmentsInput.value || "{}"); } catch (error) { assignments = {}; }
    assignments[panel.dataset.panelId] = toPost;
    assignmentsInput.value = JSON.stringify(assignments);
    panel.dataset.post = toPost;
    panel.querySelector(".panel-title span").textContent = "Loading post " + toPost + "...";
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder";
    placeholder.textContent = "Caching 5 minutes from " + toPost + "...";
    panel.querySelector(".plot-body").replaceChildren(placeholder);
    panel.querySelector(".audio-button").disabled = true;
    const parameters = new URLSearchParams(new FormData(document.getElementById("plot-form")));
    parameters.set("stream", currentAudioStream || "");
    parameters.set("offset", String(currentOffset));
    parameters.set("panel_id", panel.dataset.panelId);
    parameters.set("to_post", toPost);
    window.history.replaceState(null, "", "/?" + new URLSearchParams(new FormData(document.getElementById("plot-form"))).toString());
    openPlotStream("/reassign-stream?" + parameters.toString(), false);
    if (latestPoints) drawConnections(latestPoints);
}
function showPanelMessage(message) {
    const panel = document.querySelector("#panel-" + (message.panel_id || message.audiomoth));
    if (!panel) return;
    const body = panel.querySelector(".plot-body");
    const button = panel.querySelector(".audio-button");
    if (message.title) panel.querySelector(".panel-title span").textContent = message.title;
    if (message.post) panel.dataset.post = message.post;
    body.replaceChildren();
    if (message.image) {
        const image = document.createElement("img");
        image.src = message.image;
        image.alt = message.title || "Audio spectrogram";
        body.appendChild(image);
        button.dataset.audioUrl = message.audio_url;
        button.disabled = !message.audio_url;
    } else {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder";
        placeholder.textContent = message.error || "Recording unavailable";
        body.appendChild(placeholder);
        button.disabled = true;
    }
}
function stopActiveAudio() {
    if (activeAudio) activeAudio.pause();
    if (activeAudioButton) {
        activeAudioButton.textContent = "▶";
        activeAudioButton.classList.remove("playing");
    }
    activeAudio = null;
    activeAudioButton = null;
}
function toggleAudio(button) {
    if (activeAudioButton === button && activeAudio) {
        if (activeAudio.paused) {
            activeAudio.play();
            button.textContent = "❚❚";
            button.classList.add("playing");
        } else {
            activeAudio.pause();
            button.textContent = "▶";
            button.classList.remove("playing");
        }
        return;
    }
    stopActiveAudio();
    const audio = new Audio(button.dataset.audioUrl);
    activeAudio = audio;
    activeAudioButton = button;
    button.textContent = "❚❚";
    button.classList.add("playing");
    audio.addEventListener("ended", stopActiveAudio);
    audio.addEventListener("error", function () {
        stopActiveAudio();
        document.getElementById("status").textContent = "This browser could not play the selected WAV segment.";
    });
    audio.play().catch(function (error) {
        stopActiveAudio();
        document.getElementById("status").textContent = "Audio playback failed: " + error.message;
    });
}
function openPlotStream(url, clearPanels) {
    const plotButton = document.getElementById("plot-button");
    const backwardButton = document.getElementById("backward-button");
    const forwardButton = document.getElementById("forward-button");
    const status = document.getElementById("status");
    plotButton.disabled = true;
    backwardButton.disabled = true;
    forwardButton.disabled = true;
    stopActiveAudio();
    if (activePlotStream) activePlotStream.close();
    const operationNumber = ++plotOperationNumber;
    if (clearPanels) {
        document.querySelectorAll(".plot-body").forEach(function (body) {
            const placeholder = document.createElement("div");
            placeholder.className = "placeholder";
            placeholder.textContent = "Caching 5 minutes...";
            body.replaceChildren(placeholder);
        });
        document.querySelectorAll(".audio-button").forEach(function (audioButton) {
            audioButton.disabled = true;
            delete audioButton.dataset.audioUrl;
        });
    }
    const events = new EventSource(url);
    activePlotStream = events;
    let completed = false;
    events.onmessage = function (event) {
        if (operationNumber !== plotOperationNumber) return;
        const message = JSON.parse(event.data);
        if (message.type === "panel") showPanelMessage(message);
        if (message.type === "session") {
            currentAudioStream = message.stream_id;
            currentOffset = Number(message.offset);
        }
        if (message.type === "status" || message.type === "done" || message.type === "error") status.textContent = message.text;
        if (message.type === "done" || message.type === "error") {
            completed = true;
            if (message.stream_id) currentAudioStream = message.stream_id;
            if (message.offset !== undefined) currentOffset = Number(message.offset);
            events.close();
            activePlotStream = null;
            plotButton.disabled = false;
            updateNavigationButtons();
        }
    };
    events.onerror = function () {
        if (operationNumber !== plotOperationNumber || completed) return;
        status.textContent = "The plotting stream disconnected before it finished.";
        events.close();
        activePlotStream = null;
        plotButton.disabled = false;
        updateNavigationButtons();
    };
}
function startLoading(event) {
    const parameters = new URLSearchParams(new FormData(event.currentTarget));
    currentAudioStream = null;
    currentOffset = 0;
    document.getElementById("status").textContent = "Caching five minutes from OSN... Watch the terminal for progress.";
    window.history.replaceState(null, "", "/?" + parameters.toString());
    openPlotStream("/plot-stream?" + parameters.toString(), true);
}
function refreshCurrentSpectrograms() {
    const duration = selectedDisplayDuration();
    const status = document.getElementById("status");
    if (!Number.isFinite(duration) || duration < 0.1 || duration > 60) {
        status.textContent = "Duration must be between 0.1 and 60 seconds.";
        return;
    }
    if (!currentAudioStream) return;
    currentOffset = Math.min(currentOffset, maximumOffset());
    const parameters = new URLSearchParams(new FormData(document.getElementById("plot-form")));
    parameters.set("stream", currentAudioStream);
    parameters.set("offset", String(currentOffset));
    status.textContent = "Changing the displayed duration to " + duration + " seconds...";
    openPlotStream("/scroll-stream?" + parameters.toString(), false);
}
function stepSpectrograms(direction) {
    const step = Number(document.getElementById("scroll-step").value);
    const duration = selectedDisplayDuration();
    const status = document.getElementById("status");
    if (!Number.isFinite(step) || step <= 0) {
        status.textContent = "Step must be greater than zero.";
        return;
    }
    if (!currentAudioStream) {
        status.textContent = "Load five minutes of audio before scrolling.";
        return;
    }
    if (!Number.isFinite(duration) || duration < 0.1 || duration > 60) {
        status.textContent = "Duration must be between 0.1 and 60 seconds.";
        return;
    }
    const nextOffset = Math.max(0, Math.min(currentOffset + direction * step, maximumOffset()));
    if (nextOffset === currentOffset) {
        status.textContent = direction > 0 ? "The display is already at the end of the five-minute cache." : "The display is already at the beginning of the five-minute cache.";
        return;
    }
    const parameters = new URLSearchParams(new FormData(document.getElementById("plot-form")));
    parameters.set("stream", currentAudioStream);
    parameters.set("offset", String(nextOffset));
    status.textContent = "Requesting seconds " + nextOffset + "–" + (nextOffset + duration) + "...";
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;    openPlotStream("/scroll-stream?" + parameters.toString(), false);
}
function drawConnections(points) {
    latestPoints = points;
    const frame = document.getElementById("frame");
    const iframe = document.getElementById("map-frame");
    const svg = document.getElementById("connections");
    const frameRect = frame.getBoundingClientRect();
    const mapRect = iframe.getBoundingClientRect();
    svg.setAttribute("viewBox", "0 0 " + frame.offsetWidth + " " + frame.offsetHeight);
    svg.replaceChildren();
    document.querySelectorAll(".spectrogram").forEach(function (panel) {
        const point = points[panel.dataset.post];
        if (!point) return;
        const rect = panel.getBoundingClientRect();
        const edge = panel.dataset.edge;
        let x = rect.left - frameRect.left + rect.width / 2;
        let y = rect.top - frameRect.top + rect.height / 2;
        if (edge === "top") y = rect.bottom - frameRect.top;
        if (edge === "bottom") y = rect.top - frameRect.top;
        if (edge === "left") x = rect.right - frameRect.left;
        if (edge === "right") x = rect.left - frameRect.left;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", x); line.setAttribute("y1", y);
        line.setAttribute("x2", mapRect.left - frameRect.left + point[0]);
        line.setAttribute("y2", mapRect.top - frameRect.top + point[1]);
        line.setAttribute("stroke", "#efff38"); line.setAttribute("stroke-width", "2.5");
        line.setAttribute("stroke-linecap", "round"); line.setAttribute("opacity", ".95");
        svg.appendChild(line);
    });
}
window.addEventListener("message", function (event) {
    if (event.origin !== window.location.origin || !event.data) return;
    if (event.data.type === "folium-map-points") drawConnections(event.data.points);
    if (event.data.type === "reassign-post") reassignPost(event.data.fromPost, event.data.toPost);
});
window.addEventListener("load", function () {
    equalizePanelSizes();
    if (latestPoints) requestAnimationFrame(function () { drawConnections(latestPoints); });
});
window.addEventListener("resize", function () {
    equalizePanelSizes();
    if (latestPoints) requestAnimationFrame(function () { drawConnections(latestPoints); });
});
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    application: DeploymentWebApplication
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/map":
            self._respond(self.application.map_html)
            return
        if parsed.path == "/audio":
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            self._respond_audio(query.get("stream", ""), query.get("audiomoth", ""), query.get("offset", "0"), query.get("duration", str(DISPLAY_DURATION_SECONDS)))
            return
        if parsed.path not in {"/", "/plot-stream", "/scroll-stream", "/reassign-stream"}:
            self.send_error(404)
            return
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        parameters = {
            "start": query.get("start", "2026-08-20 20:01:00"),
            "scroll_step": query.get("scroll_step", "5.0"),
            "display_duration": query.get("display_duration", str(DISPLAY_DURATION_SECONDS)),
            "assignments": query.get("assignments", "{}"),
            "nfft": query.get("nfft", "1024"),
            "utc_offset": query.get("utc_offset", "-7"),
            "layout": query.get("layout", "example"),
            "osn_prefix": query.get("osn_prefix", self.application.osn_prefix),
        }
        if parsed.path == "/plot-stream":
            self._stream_plot(parameters)
            return
        if parsed.path == "/scroll-stream":
            parameters["stream"] = query.get("stream", "")
            parameters["offset"] = query.get("offset", "0")
            self._stream_scroll(parameters)
            return
        if parsed.path == "/reassign-stream":
            parameters["stream"] = query.get("stream", "")
            parameters["offset"] = query.get("offset", "0")
            parameters["panel_id"] = query.get("panel_id", "")
            parameters["to_post"] = query.get("to_post", "")
            self._stream_reassignment(parameters)
            return
        status = f"Loaded {len(self.application.recorders)} AudioMoths at {len(self.application.posts)} posts. Drag a red post marker onto another post to reassign its spectrogram panel."
        self._respond(self.application.render_page(parameters, status=status))

    def _stream_plot(self, parameters: dict[str, str]) -> None:
        self._stream_events(lambda send: self.application.stream_plot(parameters, send))

    def _stream_scroll(self, parameters: dict[str, str]) -> None:
        self._stream_events(lambda send: self.application.stream_scroll(parameters, send))

    def _stream_reassignment(self, parameters: dict[str, str]) -> None:
        self._stream_events(lambda send: self.application.stream_reassignment(parameters, send))

    def _stream_events(self, plotter) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(message: dict[str, object]) -> None:
            self.wfile.write(("data: " + json.dumps(message, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        try:
            plotter(send)
        except (BrokenPipeError, ConnectionResetError):
            print("[plot] Browser disconnected before plotting finished.", flush=True)
        except Exception as exc:
            print(f"[plot] ERROR: {exc}", flush=True)
            try:
                send({"type": "error", "text": f"Plot failed: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.close_connection = True

    def _respond(self, content: str) -> None:
        payload = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_audio(self, stream_id: str, audiomoth: str, offset: str, duration: str) -> None:
        try:
            payload = self.application.audio_wav(stream_id, audiomoth, float(offset), float(duration))
        except (KeyError, ValueError) as exc:
            self.send_error(404, str(exc))
            return
        start = 0
        end = len(payload) - 1
        status = 200
        range_header = self.headers.get("Range", "")
        range_match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        if range_match:
            if range_match.group(1):
                start = int(range_match.group(1))
                end = min(int(range_match.group(2)), end) if range_match.group(2) else end
            elif range_match.group(2):
                start = max(len(payload) - int(range_match.group(2)), 0)
            if start > end or start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.end_headers()
                return
            status = 206
        response_payload = payload[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(response_payload)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
        self.end_headers()
        self.wfile.write(response_payload)

    def log_message(self, format_string: str, *args) -> None:
        if self.path not in {"/map", "/favicon.ico"}:
            super().log_message(format_string, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display an interactive Folium STF deployment map framed by OSN spectrograms")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--osn-prefix", default=DEFAULT_OSN_PREFIX)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        recorders, posts = read_deployment_csv(args.csv)
        map_html = build_folium_map(posts)
        layout = example_layout(recorders)
        print(f"Validated {len(recorders)} AudioMoths, {len(posts)} posts, {sum(len(values) for values in layout.values())} example panels, and {len(map_html):,} bytes of Folium HTML")
        return
    application = DeploymentWebApplication(args.csv, args.osn_prefix)
    RequestHandler.application = application
    server = ThreadingHTTPServer(("127.0.0.1", args.port), RequestHandler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[server] Open {url}", flush=True)
    print("[server] Press Control-C in this terminal to stop the application.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
