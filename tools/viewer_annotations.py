"""Gradio CSV annotation viewer: reference overlay (read-only) + editable OBBs."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from annotation_csv import (
    AnnotationRecord,
    image_path_for_id,
    list_images_from_annotations,
    load_annotation_csv,
    next_ann_id,
    save_annotation_csv,
)
from oriented_det import RBox

# Reuse slider helpers from the main viewer when launched via app.py
try:
    from app import _slider_index, _to_float
except ImportError:
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _slider_index(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


REF_COLOR_BGR = (0, 200, 0)
EDIT_COLOR_BGR = (0, 0, 255)
SELECT_COLOR_BGR = (0, 255, 255)


def _box_label(rec: AnnotationRecord) -> str:
    rb = rec.rbox
    return f"{rec.ann_id}: {rec.class_name} @ ({rb.cx:.0f},{rb.cy:.0f})"


def _draw_rboxes(
    canvas_bgr: np.ndarray,
    records: List[AnnotationRecord],
    color_bgr: Tuple[int, int, int],
    *,
    selected_id: Optional[str] = None,
    fill_alpha: float = 0.0,
    show_labels: bool = True,
    dash_selected: bool = False,
) -> None:
    if not records:
        return
    overlay = canvas_bgr.copy() if fill_alpha > 0 else None
    for rec in records:
        points = np.array(
            [list(p) for p in rec.rbox.to_polygon().points],
            dtype=np.int32,
        )
        is_selected = selected_id is not None and rec.ann_id == selected_id
        thickness = 3 if is_selected else 2
        color = SELECT_COLOR_BGR if is_selected else color_bgr
        if overlay is not None and fill_alpha > 0:
            cv2.fillPoly(overlay, [points], color)
        if dash_selected and not is_selected:
            cv2.polylines(canvas_bgr, [points], isClosed=True, color=color, thickness=1)
        else:
            cv2.polylines(canvas_bgr, [points], isClosed=True, color=color, thickness=thickness)
        if show_labels:
            label = _box_label(rec)
            tx = int(np.min(points[:, 0]))
            ty = max(12, int(np.min(points[:, 1])) - 4)
            cv2.putText(
                canvas_bgr,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if overlay is not None and fill_alpha > 0:
        cv2.addWeighted(overlay, fill_alpha, canvas_bgr, 1.0 - fill_alpha, 0, dst=canvas_bgr)


def _apply_zoom(img_rgb: np.ndarray, scale: float) -> Image.Image:
    if scale == 1.0:
        return Image.fromarray(img_rgb, "RGB")
    h, w = img_rgb.shape[:2]
    resized = cv2.resize(
        img_rgb,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    return Image.fromarray(resized, "RGB")


class CsvAnnotationEditor:
    """In-memory editable CSV annotations with optional read-only reference layer."""

    def __init__(
        self,
        *,
        data_root: str,
        annotations_csv: str,
        reference_csv: Optional[str] = None,
        images_dir: str = "images",
    ):
        self.data_root = Path(data_root)
        self.csv_path = Path(annotations_csv)
        self.images_dir = images_dir
        self.editable = load_annotation_csv(self.csv_path)
        self.reference: Optional[Dict[str, List[AnnotationRecord]]] = None
        if reference_csv:
            self.reference = load_annotation_csv(Path(reference_csv))
        self.image_ids = list_images_from_annotations(
            self.editable,
            self.data_root,
            images_dir=images_dir,
        )
        if self.reference:
            for image_id in list_images_from_annotations(
                self.reference,
                self.data_root,
                images_dir=images_dir,
            ):
                if image_id not in self.image_ids:
                    self.image_ids.append(image_id)
        self.sorted_indices = list(range(len(self.image_ids)))
        self.current_sort = "no order"
        self.dirty = False
        total = sum(len(v) for v in self.editable.values())
        ref_total = sum(len(v) for v in self.reference.values()) if self.reference else 0
        print(f"CSV editor: {len(self.image_ids)} images, {total} editable boxes", end="")
        if self.reference is not None:
            print(f", {ref_total} reference boxes (read-only)")
        else:
            print()

    def image_id_at(self, display_idx: int) -> str:
        idx = self.sorted_indices[display_idx]
        return self.image_ids[idx]

    def _records_for(self, image_id: str, layer: str) -> List[AnnotationRecord]:
        if layer == "reference":
            if not self.reference:
                return []
            return list(self.reference.get(image_id, []))
        return list(self.editable.get(image_id, []))

    def box_choices(self, display_idx: int, layer: str) -> List[Tuple[str, str]]:
        image_id = self.image_id_at(display_idx)
        recs = self._records_for(image_id, layer)
        return [(_box_label(r), r.ann_id) for r in recs]

    def get_record(self, display_idx: int, ann_id: str) -> Optional[AnnotationRecord]:
        image_id = self.image_id_at(display_idx)
        for rec in self.editable.get(image_id, []):
            if rec.ann_id == ann_id:
                return rec
        return None

    def get_reference_record(self, display_idx: int, ann_id: str) -> Optional[AnnotationRecord]:
        if not self.reference:
            return None
        image_id = self.image_id_at(display_idx)
        for rec in self.reference.get(image_id, []):
            if rec.ann_id == ann_id:
                return rec
        return None

    def apply_sorting(self, sort_mode: str) -> None:
        self.current_sort = sort_mode
        if sort_mode == "no order":
            self.sorted_indices = list(range(len(self.image_ids)))
            return
        if sort_mode == "boxes asc":
            pairs = [
                (i, len(self.editable.get(self.image_ids[i], [])))
                for i in range(len(self.image_ids))
            ]
            pairs.sort(key=lambda x: x[1])
            self.sorted_indices = [i for i, _ in pairs]
        elif sort_mode == "boxes desc":
            pairs = [
                (i, len(self.editable.get(self.image_ids[i], [])))
                for i in range(len(self.image_ids))
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.sorted_indices = [i for i, _ in pairs]
        elif sort_mode == "ref mismatch" and self.reference is not None:
            mismatches = []
            for i, image_id in enumerate(self.image_ids):
                e = len(self.editable.get(image_id, []))
                r = len(self.reference.get(image_id, []))
                mismatches.append((abs(e - r), i))
            mismatches.sort(key=lambda x: (-x[0], x[1]))
            self.sorted_indices = [i for _, i in mismatches]
        else:
            self.sorted_indices = list(range(len(self.image_ids)))

    def render(
        self,
        display_idx: int,
        zoom_scale: float,
        view_mode: str,
        selected_ann_id: Optional[str] = None,
        show_labels: bool = True,
    ) -> Image.Image:
        if display_idx < 0 or display_idx >= len(self.sorted_indices):
            display_idx = 0
        image_id = self.image_id_at(display_idx)
        img_path = image_path_for_id(self.data_root, image_id, self.images_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            return Image.new("RGB", (800, 600), color=(200, 80, 80))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if view_mode == "image":
            return _apply_zoom(img_rgb, zoom_scale)

        out_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
        draw_ref = view_mode in ("ref_editable", "ref") and self.reference is not None
        draw_edit = view_mode in ("ref_editable", "editable")

        if draw_ref:
            _draw_rboxes(
                out_bgr,
                self._records_for(image_id, "reference"),
                REF_COLOR_BGR,
                fill_alpha=0.15,
                show_labels=show_labels,
            )
        if draw_edit:
            _draw_rboxes(
                out_bgr,
                self._records_for(image_id, "editable"),
                EDIT_COLOR_BGR,
                selected_id=selected_ann_id,
                show_labels=show_labels,
            )

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return _apply_zoom(out_rgb, zoom_scale)

    def update_box(
        self,
        display_idx: int,
        ann_id: str,
        class_name: str,
        cx: float,
        cy: float,
        width: float,
        height: float,
        angle_deg: float,
    ) -> str:
        rec = self.get_record(display_idx, ann_id)
        if rec is None:
            return "Box not found."
        rec.class_name = class_name.strip() or rec.class_name
        rec.rbox = RBox(cx, cy, max(width, 1.0), max(height, 1.0), np.radians(angle_deg))
        rec.source = "manual"
        self.dirty = True
        return f"Updated box {ann_id}."

    def delete_box(self, display_idx: int, ann_id: str) -> str:
        image_id = self.image_id_at(display_idx)
        recs = self.editable.get(image_id, [])
        new_recs = [r for r in recs if r.ann_id != ann_id]
        if len(new_recs) == len(recs):
            return "Box not found."
        if new_recs:
            self.editable[image_id] = new_recs
        else:
            self.editable.pop(image_id, None)
        self.dirty = True
        return f"Deleted box {ann_id}."

    def add_box(
        self,
        display_idx: int,
        class_name: str,
        cx: float,
        cy: float,
        width: float,
        height: float,
        angle_deg: float,
    ) -> Tuple[str, str]:
        image_id = self.image_id_at(display_idx)
        ann_id = next_ann_id(self.editable)
        rec = AnnotationRecord(
            ann_id=ann_id,
            image_id=image_id,
            class_name=class_name.strip() or "object",
            rbox=RBox(cx, cy, max(width, 1.0), max(height, 1.0), np.radians(angle_deg)),
            source="manual",
        )
        self.editable.setdefault(image_id, []).append(rec)
        self.dirty = True
        return f"Added box {ann_id}.", ann_id

    def copy_reference_to_editable(
        self,
        display_idx: int,
        ref_ann_id: str,
        edit_ann_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        ref = self.get_reference_record(display_idx, ref_ann_id)
        if ref is None:
            return "Reference box not found.", edit_ann_id
        image_id = self.image_id_at(display_idx)
        if edit_ann_id:
            rec = self.get_record(display_idx, edit_ann_id)
            if rec is None:
                return "Editable box not found.", edit_ann_id
            rec.rbox = deepcopy(ref.rbox)
            rec.class_name = ref.class_name
            rec.source = "copied_from_reference"
            self.dirty = True
            return f"Copied reference {ref_ann_id} → editable {edit_ann_id}.", edit_ann_id
        ann_id = next_ann_id(self.editable)
        rec = AnnotationRecord(
            ann_id=ann_id,
            image_id=image_id,
            class_name=ref.class_name,
            rbox=deepcopy(ref.rbox),
            source="copied_from_reference",
        )
        self.editable.setdefault(image_id, []).append(rec)
        self.dirty = True
        return f"Added editable box {ann_id} from reference {ref_ann_id}.", ann_id

    def save(self) -> str:
        save_annotation_csv(self.csv_path, self.editable)
        self.dirty = False
        return f"Saved {self.csv_path}"

    def status_line(self, display_idx: int) -> str:
        image_id = self.image_id_at(display_idx)
        n_edit = len(self.editable.get(image_id, []))
        n_ref = len(self.reference.get(image_id, [])) if self.reference else 0
        dirty = " *(unsaved)*" if self.dirty else ""
        idx_str = f"{display_idx + 1} / {len(self.sorted_indices)}"
        parts = [f"**Image {idx_str}:** `{image_id}`", f"editable: **{n_edit}**"]
        if self.reference is not None:
            parts.append(f"reference: **{n_ref}**")
        parts.append(dirty)
        return " · ".join(parts)


def _record_fields(rec: Optional[AnnotationRecord]) -> Tuple[str, float, float, float, float, float]:
    if rec is None:
        return "object", 128.0, 128.0, 80.0, 40.0, 0.0
    rb = rec.rbox
    return (
        rec.class_name,
        float(rb.cx),
        float(rb.cy),
        float(rb.width),
        float(rb.height),
        float(np.degrees(rb.angle)),
    )


def create_csv_annotation_app(
    data_root: str,
    annotations_csv: str,
    reference_csv: Optional[str] = None,
    images_dir: str = "images",
) -> gr.Blocks:
    editor = CsvAnnotationEditor(
        data_root=data_root,
        annotations_csv=annotations_csv,
        reference_csv=reference_csv,
        images_dir=images_dir,
    )

    def _refresh_view(
        display_idx: int,
        zoom_str: str,
        sort_mode: str,
        view_mode: str,
        selected_ann_id: Optional[str],
        show_labels: bool,
    ):
        display_idx = _slider_index(display_idx, 0)
        if editor.current_sort != sort_mode:
            editor.apply_sorting(str(sort_mode))
        zoom = float(str(zoom_str or "2x").replace("x", ""))
        img_combo = editor.render(
            display_idx,
            zoom,
            view_mode,
            selected_ann_id=selected_ann_id or None,
            show_labels=bool(show_labels),
        )
        img_edit = editor.render(display_idx, zoom, "editable", selected_ann_id, show_labels)
        img_ref = editor.render(display_idx, zoom, "ref", show_labels=show_labels)
        img_plain = editor.render(display_idx, zoom, "image")
        status = editor.status_line(display_idx)
        edit_choices = editor.box_choices(display_idx, "editable")
        ref_choices = editor.box_choices(display_idx, "reference")
        edit_ids = [c[1] for c in edit_choices]
        ref_ids = [c[1] for c in ref_choices]
        sel = selected_ann_id if selected_ann_id in edit_ids else (edit_ids[0] if edit_ids else None)
        rec = editor.get_record(display_idx, sel) if sel else None
        class_name, cx, cy, w, h, ang = _record_fields(rec)
        return (
            img_combo,
            img_edit,
            img_ref,
            img_plain,
            status,
            edit_choices,
            sel,
            ref_choices,
            ref_ids[0] if ref_ids else None,
            class_name,
            cx,
            cy,
            w,
            h,
            ang,
            display_idx,
            display_idx,
        )

    def _on_select_box(display_idx, zoom_str, sort_mode, view_mode, ann_id, show_labels):
        display_idx = _slider_index(display_idx, 0)
        rec = editor.get_record(display_idx, ann_id) if ann_id else None
        class_name, cx, cy, w, h, ang = _record_fields(rec)
        zoom = float(str(zoom_str or "2x").replace("x", ""))
        img_combo = editor.render(
            display_idx,
            zoom,
            view_mode,
            selected_ann_id=ann_id,
            show_labels=bool(show_labels),
        )
        return img_combo, class_name, cx, cy, w, h, ang

    def _dropdown_updates(edit_choices, edit_val, ref_choices, ref_val):
        return (
            gr.update(choices=edit_choices, value=edit_val),
            gr.update(choices=ref_choices, value=ref_val),
        )

    def _pack_refresh(out):
        imgs = out[:5]
        dd = _dropdown_updates(out[5], out[6], out[7], out[8])
        return imgs + dd + out[9:]

    def _apply_edit(display_idx, zoom_str, sort_mode, view_mode, ann_id, show_labels, cls, cx, cy, w, h, ang):
        msg = editor.update_box(display_idx, ann_id, cls, cx, cy, w, h, ang)
        out = _pack_refresh(_refresh_view(display_idx, zoom_str, sort_mode, view_mode, ann_id, show_labels))
        return (*out, msg)

    def _delete(display_idx, zoom_str, sort_mode, view_mode, ann_id, show_labels):
        msg = editor.delete_box(display_idx, ann_id)
        out = _pack_refresh(_refresh_view(display_idx, zoom_str, sort_mode, view_mode, None, show_labels))
        return (*out, msg)

    def _add(display_idx, zoom_str, sort_mode, view_mode, show_labels, cls, cx, cy, w, h, ang):
        msg, new_id = editor.add_box(display_idx, cls, cx, cy, w, h, ang)
        out = _pack_refresh(_refresh_view(display_idx, zoom_str, sort_mode, view_mode, new_id, show_labels))
        return (*out, msg)

    def _copy_ref(display_idx, zoom_str, sort_mode, view_mode, edit_id, ref_id, show_labels):
        msg, new_id = editor.copy_reference_to_editable(display_idx, ref_id, edit_id)
        out = _pack_refresh(_refresh_view(display_idx, zoom_str, sort_mode, view_mode, new_id, show_labels))
        return (*out, msg)

    def _save():
        return editor.save()

    init = _refresh_view(0, "2x", "no order", "ref_editable", None, True)

    sort_choices = ["no order", "boxes asc", "boxes desc"]
    if editor.reference is not None:
        sort_choices.append("ref mismatch")

    with gr.Blocks(title="CSV Annotation Editor") as app:
        gr.Markdown(
            "# CSV annotation editor\n"
            "Green = read-only reference. Red = editable oriented boxes (yellow = selected). "
            "Changes are kept in memory until **Save CSV**."
        )
        status_md = gr.Markdown(value=init[4])
        with gr.Tabs():
            with gr.Tab("Reference + Editable"):
                tab_combo = gr.Image(value=init[0], label="Reference (green) + editable (red)", height=700)
            with gr.Tab("Editable only"):
                tab_edit = gr.Image(value=init[1], label="Editable boxes", height=700)
            with gr.Tab("Reference only"):
                tab_ref = gr.Image(value=init[2], label="Reference boxes", height=700)
            with gr.Tab("Original image"):
                tab_image = gr.Image(value=init[3], label="Image", height=700)

        active_view = gr.State("ref_editable")

        with gr.Row():
            first_btn = gr.Button("First")
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
            last_btn = gr.Button("Last")

        with gr.Row():
            image_idx = gr.Slider(0, max(0, len(editor.image_ids) - 1), value=0, step=1, label="Image index")
            image_idx_state = gr.State(0)
            zoom_scale = gr.Radio(["1x", "2x", "4x"], value="2x", label="Zoom")
            sort_mode = gr.Dropdown(sort_choices, value="no order", label="Sort")
            show_labels = gr.Checkbox(value=True, label="Show labels")

        with gr.Row():
            edit_box_dd = gr.Dropdown(choices=init[5], value=init[6], label="Editable box")
            ref_box_dd = gr.Dropdown(choices=init[7], value=init[8], label="Reference box")

        with gr.Row():
            class_name = gr.Textbox(value=init[9], label="Class")
            cx_in = gr.Number(value=init[10], label="cx (px)")
            cy_in = gr.Number(value=init[11], label="cy (px)")
            width_in = gr.Number(value=init[12], label="width (px)")
            height_in = gr.Number(value=init[13], label="height (px)")
            angle_in = gr.Number(value=init[14], label="angle (deg)")

        with gr.Row():
            apply_btn = gr.Button("Apply changes", variant="primary")
            add_btn = gr.Button("Add box")
            delete_btn = gr.Button("Delete selected", variant="stop")
            copy_ref_btn = gr.Button("Copy reference → editable")
            save_btn = gr.Button("Save CSV", variant="primary")

        action_msg = gr.Markdown("")

        refresh_inputs = [
            image_idx_state,
            zoom_scale,
            sort_mode,
            active_view,
            edit_box_dd,
            show_labels,
        ]
        refresh_outputs = [
            tab_combo,
            tab_edit,
            tab_ref,
            tab_image,
            status_md,
            edit_box_dd,
            edit_box_dd,
            ref_box_dd,
            ref_box_dd,
            class_name,
            cx_in,
            cy_in,
            width_in,
            height_in,
            angle_in,
            image_idx,
            image_idx_state,
        ]

        def _nav(first: bool, last: bool, delta: int, *args):
            idx = _slider_index(args[0], 0)
            if first:
                idx = 0
            elif last:
                idx = len(editor.image_ids) - 1
            else:
                idx = max(0, min(len(editor.image_ids) - 1, idx + delta))
            return _pack_refresh(_refresh_view(idx, *args[1:]))

        nav_inputs = [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels]

        first_btn.click(lambda *a: _nav(True, False, 0, *a), nav_inputs, refresh_outputs)
        prev_btn.click(lambda *a: _nav(False, False, -1, *a), nav_inputs, refresh_outputs)
        next_btn.click(lambda *a: _nav(False, False, 1, *a), nav_inputs, refresh_outputs)
        last_btn.click(lambda *a: _nav(False, True, 0, *a), nav_inputs, refresh_outputs)

        image_idx.change(
            lambda idx, *rest: _pack_refresh(_refresh_view(idx, *rest)),
            [image_idx, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            refresh_outputs,
        )
        zoom_scale.change(
            lambda idx, *rest: _pack_refresh(_refresh_view(idx, *rest)),
            [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            refresh_outputs,
        )
        sort_mode.change(
            lambda idx, *rest: _pack_refresh(_refresh_view(idx, *rest)),
            [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            refresh_outputs,
        )
        show_labels.change(
            lambda idx, *rest: _pack_refresh(_refresh_view(idx, *rest)),
            [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            refresh_outputs,
        )

        edit_box_dd.change(
            _on_select_box,
            [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            [tab_combo, class_name, cx_in, cy_in, width_in, height_in, angle_in],
        )

        apply_btn.click(
            _apply_edit,
            [
                image_idx_state,
                zoom_scale,
                sort_mode,
                active_view,
                edit_box_dd,
                show_labels,
                class_name,
                cx_in,
                cy_in,
                width_in,
                height_in,
                angle_in,
            ],
            refresh_outputs + [action_msg],
        )
        delete_btn.click(
            _delete,
            [image_idx_state, zoom_scale, sort_mode, active_view, edit_box_dd, show_labels],
            refresh_outputs + [action_msg],
        )
        add_btn.click(
            _add,
            [
                image_idx_state,
                zoom_scale,
                sort_mode,
                active_view,
                show_labels,
                class_name,
                cx_in,
                cy_in,
                width_in,
                height_in,
                angle_in,
            ],
            refresh_outputs + [action_msg],
        )
        copy_ref_btn.click(
            _copy_ref,
            [
                image_idx_state,
                zoom_scale,
                sort_mode,
                active_view,
                edit_box_dd,
                ref_box_dd,
                show_labels,
            ],
            refresh_outputs + [action_msg],
        )
        save_btn.click(_save, [], action_msg)

        tab_combo.select(lambda: "ref_editable", outputs=active_view)
        tab_edit.select(lambda: "editable", outputs=active_view)
        tab_ref.select(lambda: "ref", outputs=active_view)
        tab_image.select(lambda: "image", outputs=active_view)

    return app
