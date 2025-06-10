import pandas
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import multiprocessing
import logging
import utils

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

MUST_CONTAIN = [
    "SeriesInstanceUID",
    "StudyDate",
    "CoordX",
    "CoordY",
    "CoordZ",
    "PatientID",
    "LesionID"
]

SAVE_FORMATS = [
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
]


class NoduleExtractor:
    """
    Class to load a dataset and extract nodule patches
    and save them to disk in a format compatible with SimpleITK
    """

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        output_path: Path,
        postfix: str = "",
        patch_size: np.array = np.array([128, 128, 64]),
        save_format: str = ".nii.gz",
    ) -> None:

        self.dataset: pandas.DataFrame = pandas.read_csv(csv_path)
        self.output_path = output_path
        self.image_root = image_root
        self.postfix = postfix
        self.patch_size = np.array(patch_size)
        self.save_format = save_format

        self.output_path.mkdir(exist_ok=True, parents=True)

        assert save_format in SAVE_FORMATS, "save format not supported"

        assert np.all(
            [k in self.dataset.keys() for k in MUST_CONTAIN]
        ), f"keys missing: the CSV must contain {MUST_CONTAIN}"

        self.dataset = utils.standardize_nodule_df(self.dataset)

    def process_seriesuid(
        self,
        seriesuid: str,
    ) -> None:
        """Function to generate 3D patches around nodules for a given SeriesInstanceUID
        Expects the dataframe `dataset` to be globally present
        Args:
            seriesuid (str)
        """
        logging.info(f"Extracting nodule blocks from {seriesuid}")

        subset = self.dataset[self.dataset.SeriesInstanceUID == seriesuid]
        extent = self.patch_size // 2

        image_path = str(self.image_root / f"{seriesuid}.mha")

        # Check if nodule is already extracted
        coords = np.array(
            [
                subset.CoordX,
                subset.CoordY,
                subset.CoordZ,
            ]
        ).transpose()

        for i, coord in enumerate(coords):

            pd = subset.iloc[i]
            output_path = (
                self.output_path
                / f"{pd.AnnotationID}{self.postfix}{self.save_format}"
            )

            if Path(output_path).is_file():
                logging.info(f"{pd.NoduleID} of {seriesuid} is already extracted")

            else:
                if Path(image_path).is_file():

                    image = sitk.ReadImage(image_path)

                    pad = False

                    index_coord = np.array(image.TransformPhysicalPointToIndex(coord))
                    min_coordinate_out_of_bounds = np.any(index_coord - extent < 0)
                    max_coordinate_out_of_bounds = np.any(index_coord + extent > np.array(image.GetSize()))

                    if min_coordinate_out_of_bounds or max_coordinate_out_of_bounds:
                        pad = True

                    if pad:
                        image = sitk.ConstantPad(
                            image,
                            [int(e) for e in extent],
                            [int(e) for e in extent],
                            constant=-1024,
                        )
                        index_coord = image.TransformPhysicalPointToIndex(coord)

                    image_patch = image[
                        int(index_coord[0] - extent[0]): int(index_coord[0] + extent[0]),
                        int(index_coord[1] - extent[1]): int(index_coord[1] + extent[1]),
                        int(index_coord[2] - extent[2]): int(index_coord[2] + extent[2]),
                    ]

                    if image_patch.GetSize() == tuple(self.patch_size):
                        sitk.WriteImage(image_patch, str(output_path), True)
                    else:
                        logging.info(f"Incorrect patch size in: {seriesuid}")

                else:
                    logging.info(f"Missing mha file: {str(image_path)}")


if __name__ == "__main__":
  
    extractor = NoduleExtractor(
        csv_path=Path(
            <PATH-TO-CSV>
        ),
        image_root=Path(<PATH-TO-IMAGES>),
        output_path=Path(
            <PATH-TO-OUTPUT>
        ),
        postfix="_0000",
        save_format=".nii.gz",
    )

    pool = multiprocessing.Pool(16)
    pool.map(
        extractor.process_seriesuid,
        extractor.dataset.SeriesInstanceUID.unique(),
    )
    pool.close()
