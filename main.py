"""
Main File for Drone Image AI Analysis Pipeline
"""

import argparse
import os
import sys
import time
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    os.path.join(ROOT, "src")
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data_loader import (
    DroneImageLoader,
    load_drone_image
)

from preprocessing import ImagePreprocessor

from feature_extraction import FeatureExtractor

from model_training import (
    LandCoverClassifier,
    CNNClassifier,
    generate_labels_from_image,
    augment_patches,
    CLASS_NAMES
)

from prediction import (
    PredictionPipeline,
    run_prediction
)

from visualization import (
    generate_all_outputs
)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Drone Image AI Analysis Pipeline"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Path to input TIFF/JPG/PNG. "
            "Defaults to data/Drone_SAMPLE.tif"
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=1024,
        help=(
            "Resize image to this side length "
            "before processing"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=[
            "rf",
            "gb",
            "svm"
        ]
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay transparency [0-1]"
    )

    parser.add_argument(
        "--use_cnn",
        action="store_true",
        help=(
            "Also train CNN "
            "(requires PyTorch)"
        )
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable patch augmentation. "
            "Disabled by default."
        )
    )

    return parser.parse_args()


# ============================================================
# PRINT HELPERS
# ============================================================

def _header(text):

    width = 60

    print(
        f"\n{'=' * width}"
    )

    print(
        f"  {text}"
    )

    print(
        f"{'=' * width}"
    )


def _step(
    number,
    total,
    text
):

    print(
        f"\nStep {number}/{total}: {text}"
    )


def _done(elapsed):

    print(
        f"Done ({elapsed:.1f}s)"
    )


# ============================================================
# FIND IMAGE
# ============================================================

def _find_image(arg_path):

    candidates = []

    # User-provided path
    if arg_path:
        candidates.append(
            arg_path
        )

    # Real image
    candidates.append(
        os.path.join(
            ROOT,
            "data",
            "Drone_SAMPLE.tif"
        )
    )

    # TIFF alternative
    candidates.append(
        os.path.join(
            ROOT,
            "data",
            "Drone_SAMPLE.tiff"
        )
    )

    # Other previous names
    candidates.append(
        os.path.join(
            ROOT,
            "data",
            "drone_image.tif"
        )
    )

    candidates.append(
        os.path.join(
            ROOT,
            "data",
            "drone_image.tiff"
        )
    )

    candidates.append(
        os.path.join(
            ROOT,
            "data",
            "drone_image.jpg"
        )
    )

    for candidate in candidates:

        if candidate and os.path.exists(
            candidate
        ):

            return os.path.abspath(
                candidate
            )

    raise FileNotFoundError(
        "\nNo drone image found.\n\n"
        "Expected location:\n"
        f"{os.path.join(ROOT, 'data', 'Drone_SAMPLE.tif')}\n\n"
        "Or provide a custom path using:\n"
        "--image <path>"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    args = parse_args()

    # Create output directory
    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    TOTAL_STEPS = 8

    global_start = time.time()

    _header(
        "Drone Image AI Analysis Pipeline - Starting"
    )

    # ========================================================
    # STEP 1 - DATA LOADING
    # ========================================================

    _step(
        1,
        TOTAL_STEPS,
        "Data Loading"
    )

    t0 = time.time()

    image_path = _find_image(
        args.image
    )

    print(
        f"  Image path : {image_path}"
    )

    loader = DroneImageLoader(
        image_path,
        verbose=True
    )

    raw_img = loader.load()

    metadata = loader.metadata

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 2 - PREPROCESSING
    # ========================================================

    _step(
        2,
        TOTAL_STEPS,
        "Preprocessing"
    )

    t0 = time.time()

    preprocessor = ImagePreprocessor(
        raw_img
    )

    proc_f32 = (
        preprocessor.run_pipeline(
            target_size=(
                args.resize,
                args.resize
            ),
            denoise_method="bilateral",
            norm_method="minmax",
            do_enhance=True,
            do_sharpen=False
        )
    )

    proc_u8 = (
        preprocessor.get_uint8()
    )

    print(
        f"  Original shape : "
        f"{raw_img.shape}"
    )

    print(
        f"  Processed shape: "
        f"{proc_u8.shape}"
    )

    print(
        f"  Processed dtype : "
        f"{proc_u8.dtype}"
    )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 3 - GENERATE LABELS
    # ========================================================

    _step(
        3,
        TOTAL_STEPS,
        "Generating heuristic training labels"
    )

    t0 = time.time()

    patches, labels = (
        generate_labels_from_image(
            proc_u8,
            tile_size=args.tile_size
        )
    )

    print(
        f"  Patches generated : "
        f"{len(patches)}"
    )

    print(
        f"  Labels generated  : "
        f"{len(labels)}"
    )

    # Class distribution
    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print(
        "\n  Class distribution:"
    )

    for cls, count in zip(
        unique,
        counts
    ):

        try:
            class_name = CLASS_NAMES[
                int(cls)
            ]
        except Exception:
            class_name = str(cls)

        print(
            f"    {cls} - "
            f"{class_name}: "
            f"{count}"
        )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 4 - AUGMENTATION
    # ========================================================

    _step(
        4,
        TOTAL_STEPS,
        "Data Augmentation"
    )

    t0 = time.time()

    if args.augment:

        patches, labels = (
            augment_patches(
                patches,
                labels
            )
        )

        print(
            "  Augmentation enabled."
        )

        print(
            f"  Patches after augmentation : "
            f"{len(patches)}"
        )

    else:

        print(
            "  Augmentation disabled."
        )

        print(
            "  Using patches extracted "
            "directly from the real TIFF."
        )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 5 - FEATURE EXTRACTION
    # ========================================================

    _step(
        5,
        TOTAL_STEPS,
        "Feature Extraction"
    )

    t0 = time.time()

    extractor = FeatureExtractor(
        patch_size=args.tile_size
    )

    X = extractor.extract_batch(
        patches,
        verbose=True
    )

    y = labels

    print(
        f"  Feature matrix : "
        f"{X.shape}"
    )

    print(
        f"  Labels         : "
        f"{y.shape}"
    )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 6 - TRAINING
    # ========================================================

    _step(
        6,
        TOTAL_STEPS,
        f"Training {args.model.upper()} Classifier"
    )

    t0 = time.time()

    classifier = LandCoverClassifier(
        args.model,
        n_estimators=args.n_estimators
    )

    metrics = classifier.train(
        X,
        y,
        val_size=0.2
    )

    model_path = os.path.join(
        args.output_dir,
        "land_cover_model.pkl"
    )

    classifier.save(
        model_path
    )

    print(
        f"\n  Model saved : {model_path}"
    )

    print(
        f"  Validation accuracy : "
        f"{metrics['val_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # OPTIONAL CNN
    # --------------------------------------------------------

    if args.use_cnn:

        print(
            "\n  Training optional CNN..."
        )

        cnn = CNNClassifier(
            num_classes=len(
                CLASS_NAMES
            ),
            patch_size=args.tile_size
        )

        if cnn.model is not None:

            cnn.train(
                patches,
                labels,
                epochs=30
            )

            cnn_path = os.path.join(
                args.output_dir,
                "cnn_model.pt"
            )

            cnn.save(
                cnn_path
            )

            print(
                f"  CNN saved : {cnn_path}"
            )

        else:

            print(
                "  CNN unavailable."
            )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 7 - PREDICTION
    # ========================================================

    _step(
        7,
        TOTAL_STEPS,
        "Running Tile-by-Tile Prediction"
    )

    t0 = time.time()

    pipeline = PredictionPipeline(
        classifier,
        tile_size=args.tile_size,
        overlap=0
    )

    results = pipeline.predict(
        proc_u8
    )

    csv_path = os.path.join(
        args.output_dir,
        "predictions.csv"
    )

    pipeline.export_csv(
        csv_path
    )

    label_img = (
        pipeline.get_label_image(
            proc_u8.shape
        )
    )

    confidence_img = (
        pipeline.get_confidence_image(
            proc_u8.shape
        )
    )

    proba_img = (
        pipeline.get_proba_image(
            proc_u8.shape
        )
    )

    print(
        f"  Predictions CSV : "
        f"{csv_path}"
    )

    print(
        f"  Tiles predicted : "
        f"{len(results['records'])}"
    )

    _done(
        time.time() - t0
    )

    # ========================================================
    # STEP 8 - VISUALIZATION
    # ========================================================

    _step(
        8,
        TOTAL_STEPS,
        "Generating Visualisations"
    )

    t0 = time.time()

    saved = generate_all_outputs(
        original_image=proc_u8,
        label_image=label_img,
        confidence_image=confidence_img,
        proba_image=proba_img,
        cls_map=results[
            "classification_map"
        ],
        feature_importances=metrics.get(
            "feature_importances"
        ),
        feature_names=extractor.feature_names,
        output_dir=args.output_dir,
        alpha=args.alpha
    )

    _done(
        time.time() - t0
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = (
        time.time() - global_start
    )

    _header(
        "Pipeline Complete!"
    )

    print(
        f"  Total time        : "
        f"{elapsed:.1f}s"
    )

    print(
        f"  Image loaded      : "
        f"{metadata['file']}"
    )

    print(
        f"  Image dimensions  : "
        f"{metadata['width_px']} × "
        f"{metadata['height_px']}"
    )

    print(
        f"  Training patches  : "
        f"{len(patches)}"
    )

    print(
        f"  Feature dimensions: "
        f"{X.shape[1]}"
    )

    print(
        f"  Validation acc    : "
        f"{metrics['val_accuracy']:.4f}"
    )

    print(
        f"  Tiles predicted   : "
        f"{len(results['records'])}"
    )

    print(
        f"\n  Output directory  : "
        f"{args.output_dir}/"
    )

    print(
        f"\n  Output files:"
    )

    print(
        f"  {'-' * 46}"
    )

    all_files = sorted(
        filename
        for filename in os.listdir(
            args.output_dir
        )
        if os.path.isfile(
            os.path.join(
                args.output_dir,
                filename
            )
        )
    )

    for filename in all_files:

        path = os.path.join(
            args.output_dir,
            filename
        )

        size_kb = (
            os.path.getsize(path)
            // 1024
        )

        print(
            f"    {filename:45s} "
            f"{size_kb:6d} KB"
        )

    print(
        "\nAll done!\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()