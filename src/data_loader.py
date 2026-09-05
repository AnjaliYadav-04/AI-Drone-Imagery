"""
Drone Image Data Loader

Supports:
    - GeoTIFF / TIFF
    - JPG / JPEG
    - PNG

The loader converts the input image into:
    (H, W, 3) uint8 RGB

For GeoTIFF files, rasterio is preferred when available.
"""

import os
import numpy as np
import cv2
from PIL import Image

# Optional libraries
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


class DroneImageLoader:

    SUPPORTED_EXT = {
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".png",
    }

    def __init__(self, image_path, verbose=True):
        self.image_path = os.path.abspath(image_path)
        self.verbose = verbose

        self.image = None
        self.metadata = {}
        self.geo_metadata = {}

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(
                f"Image not found:\n{self.image_path}"
            )

        ext = os.path.splitext(self.image_path)[1].lower()

        if ext not in self.SUPPORTED_EXT:
            raise ValueError(
                f"Unsupported image format: {ext}\n"
                f"Supported formats: {sorted(self.SUPPORTED_EXT)}"
            )

    # ---------------------------------------------------------
    # MAIN LOAD FUNCTION
    # ---------------------------------------------------------

    def load(self):
        """
        Load image and return RGB uint8 array.

        Returns
        -------
        np.ndarray
            Shape: (H, W, 3)
            dtype: uint8
        """

        ext = os.path.splitext(self.image_path)[1].lower()

        if ext in {".tif", ".tiff"}:
            image = self._load_tiff()
        else:
            image = self._load_standard_image()

        image = self._to_rgb_uint8(image)

        self.image = image

        self._build_metadata()

        if self.verbose:
            self._print_metadata()

        return self.image

    # ---------------------------------------------------------
    # TIFF / GEOTIFF
    # ---------------------------------------------------------

    def _load_tiff(self):
        """
        Load TIFF / GeoTIFF.

        rasterio is preferred because it can preserve
        geospatial metadata such as CRS and transform.
        """

        # ---------------------------------------------
        # Preferred method: rasterio
        # ---------------------------------------------

        if RASTERIO_AVAILABLE:

            try:
                with rasterio.open(self.image_path) as src:

                    data = src.read()

                    # Save GeoTIFF metadata
                    self.geo_metadata = {
                        "driver": src.driver,
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "dtype": str(src.dtypes[0]),
                        "crs": str(src.crs) if src.crs else None,
                        "transform": str(src.transform),
                        "bounds": tuple(src.bounds),
                        "resolution": tuple(src.res),
                    }

                    if self.verbose:
                        print("  TIFF reader : rasterio")
                        print(f"  Bands       : {src.count}")

                        if src.crs:
                            print(f"  CRS         : {src.crs}")

                    return data

            except Exception as e:

                if self.verbose:
                    print(
                        f"  rasterio TIFF read failed: {e}"
                    )

        # ---------------------------------------------
        # Fallback: tifffile
        # ---------------------------------------------

        if TIFFFILE_AVAILABLE:

            try:

                data = tifffile.imread(self.image_path)

                if self.verbose:
                    print("  TIFF reader : tifffile")

                return data

            except Exception as e:

                if self.verbose:
                    print(
                        f"  tifffile read failed: {e}"
                    )

        # ---------------------------------------------
        # Final fallback: PIL
        # ---------------------------------------------

        try:

            image = Image.open(self.image_path)

            if self.verbose:
                print("  TIFF reader : PIL")

            return np.array(image)

        except Exception as e:

            raise RuntimeError(
                f"Unable to read TIFF file:\n"
                f"{self.image_path}\n\n"
                f"Error: {e}"
            )

    # ---------------------------------------------------------
    # JPG / PNG
    # ---------------------------------------------------------

    def _load_standard_image(self):

        image = cv2.imread(
            self.image_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read:\n"
                f"{self.image_path}"
            )

        # OpenCV BGR -> RGB
        if image.ndim == 3:

            if image.shape[2] >= 3:

                image = cv2.cvtColor(
                    image[:, :, :3],
                    cv2.COLOR_BGR2RGB
                )

        return image

    # ---------------------------------------------------------
    # CONVERT TO RGB UINT8
    # ---------------------------------------------------------

    def _to_rgb_uint8(self, arr):

        arr = np.asarray(arr)

        if arr.size == 0:
            raise ValueError("Image contains no data.")

        # Remove singleton dimensions
        arr = np.squeeze(arr)

        # -----------------------------------------------------
        # 2D grayscale
        # -----------------------------------------------------

        if arr.ndim == 2:

            arr = self._scale_to_uint8(arr)

            arr = np.stack(
                [arr, arr, arr],
                axis=-1
            )

            return arr

        # -----------------------------------------------------
        # 3D data
        # -----------------------------------------------------

        if arr.ndim == 3:

            # Detect CHW format
            if (
                arr.shape[0] in (1, 3, 4, 5, 6, 8, 10, 12)
                and arr.shape[0] < arr.shape[1]
                and arr.shape[0] < arr.shape[2]
            ):
                arr = np.transpose(
                    arr,
                    (1, 2, 0)
                )

            # -------------------------------------------------
            # Single band
            # -------------------------------------------------

            if arr.shape[2] == 1:

                band = self._scale_to_uint8(
                    arr[:, :, 0]
                )

                return np.stack(
                    [band, band, band],
                    axis=-1
                )

            # -------------------------------------------------
            # RGB
            # -------------------------------------------------

            if arr.shape[2] >= 3:

                arr = arr[:, :, :3]

                arr = self._scale_rgb_to_uint8(arr)

                return arr

        raise ValueError(
            "Unsupported image array shape: "
            f"{arr.shape}"
        )

    # ---------------------------------------------------------
    # SCALE SINGLE BAND
    # ---------------------------------------------------------

    def _scale_to_uint8(self, arr):

        arr = np.asarray(arr)

        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype(np.float32)

        finite = np.isfinite(arr)

        if not np.any(finite):
            return np.zeros(
                arr.shape,
                dtype=np.uint8
            )

        valid = arr[finite]

        # Percentile stretch
        low = np.percentile(valid, 2)
        high = np.percentile(valid, 98)

        if high <= low:
            low = valid.min()
            high = valid.max()

        if high <= low:
            return np.zeros(
                arr.shape,
                dtype=np.uint8
            )

        arr = np.clip(
            arr,
            low,
            high
        )

        arr = (
            (arr - low)
            / (high - low)
            * 255.0
        )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

        return arr.astype(np.uint8)

    # ---------------------------------------------------------
    # SCALE RGB
    # ---------------------------------------------------------

    def _scale_rgb_to_uint8(self, arr):

        if arr.dtype == np.uint8:
            return arr

        output = np.zeros(
            arr.shape,
            dtype=np.uint8
        )

        for band in range(3):

            output[:, :, band] = (
                self._scale_to_uint8(
                    arr[:, :, band]
                )
            )

        return output

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def _build_metadata(self):

        h, w, c = self.image.shape

        self.metadata = {

            "file": os.path.basename(
                self.image_path
            ),

            "path": self.image_path,

            "width_px": int(w),

            "height_px": int(h),

            "channels": int(c),

            "dtype": str(
                self.image.dtype
            ),

            "pixel_min": int(
                self.image.min()
            ),

            "pixel_max": int(
                self.image.max()
            ),

            "pixel_mean": float(
                self.image.mean()
            ),

            "memory_mb": float(
                self.image.nbytes
                / (1024 ** 2)
            ),

        }

        # Add GeoTIFF information
        if self.geo_metadata:

            self.metadata.update(
                {
                    "crs": self.geo_metadata.get(
                        "crs"
                    ),

                    "resolution": self.geo_metadata.get(
                        "resolution"
                    ),

                    "bounds": self.geo_metadata.get(
                        "bounds"
                    ),
                }
            )

    # ---------------------------------------------------------
    # PRINT METADATA
    # ---------------------------------------------------------

    def _print_metadata(self):

        print("\nImage Metadata")
        print("-" * 50)

        print(
            f"  File       : "
            f"{self.metadata['file']}"
        )

        print(
            f"  Dimensions : "
            f"{self.metadata['width_px']} × "
            f"{self.metadata['height_px']}"
        )

        print(
            f"  Channels   : "
            f"{self.metadata['channels']}"
        )

        print(
            f"  Data type  : "
            f"{self.metadata['dtype']}"
        )

        print(
            f"  Pixel min  : "
            f"{self.metadata['pixel_min']}"
        )

        print(
            f"  Pixel max  : "
            f"{self.metadata['pixel_max']}"
        )

        print(
            f"  Pixel mean : "
            f"{self.metadata['pixel_mean']:.2f}"
        )

        print(
            f"  Memory     : "
            f"{self.metadata['memory_mb']:.2f} MB"
        )

        if self.geo_metadata:

            print("\nGeoTIFF Metadata")
            print("-" * 50)

            print(
                f"  CRS        : "
                f"{self.geo_metadata.get('crs')}"
            )

            print(
                f"  Resolution : "
                f"{self.geo_metadata.get('resolution')}"
            )

            print(
                f"  Bounds     : "
                f"{self.geo_metadata.get('bounds')}"
            )

    # ---------------------------------------------------------
    # TILING
    # ---------------------------------------------------------

    def tile_image(
        self,
        tile_size=64,
        overlap=0
    ):
        """
        Split image into tiles.

        Returns
        -------
        patches : list
        positions : list
        """

        if self.image is None:
            self.load()

        image = self.image

        h, w, _ = image.shape

        stride = tile_size - overlap

        if stride <= 0:
            raise ValueError(
                "overlap must be smaller than tile_size"
            )

        patches = []
        positions = []

        for y in range(
            0,
            h - tile_size + 1,
            stride
        ):

            for x in range(
                0,
                w - tile_size + 1,
                stride
            ):

                patch = image[
                    y:y + tile_size,
                    x:x + tile_size
                ]

                patches.append(patch)

                positions.append(
                    (x, y)
                )

        return patches, positions


# -------------------------------------------------------------
# SIMPLE FUNCTION
# -------------------------------------------------------------

def load_drone_image(image_path):

    loader = DroneImageLoader(
        image_path,
        verbose=True
    )

    return loader.load()


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    image_path = os.path.join(
        current_dir,
        "..",
        "data",
        "Drone_SAMPLE.tif"
    )

    image_path = os.path.abspath(
        image_path
    )

    print(
        f"Testing image loader:\n"
        f"{image_path}"
    )

    loader = DroneImageLoader(
        image_path,
        verbose=True
    )

    image = loader.load()

    print("\nLoader test successful.")

    print(
        f"Final image shape : "
        f"{image.shape}"
    )

    print(
        f"Final dtype       : "
        f"{image.dtype}"
    )