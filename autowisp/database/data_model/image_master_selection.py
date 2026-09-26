"""Define the ImageMasterSelection table for the pipeline."""

from sqlalchemy import Column, Integer, String, ForeignKey

from autowisp.database.data_model.base import DataModelSubBase

__all__ = ["ImageMasterSelection"]


class ImageMasterSelection(DataModelSubBase):
    """Records the resolved master file selected for each image/channel/type.

    Holds the single photometric reference of every image/channel
    fit_magnitudes sees. Rows are written when a reference is registered in
    the BUI, binding the qualifying images, and by the pipeline before
    magnitude fitting, for images not yet bound: to a suitable reference
    within ``max_photref_separation`` if that is finite, otherwise to the one
    the condition expressions select. All go through
    :func:`autowisp.database.image_processing.record_photref_bindings`.
    Downstream steps (create_lightcurves, epd, tfa) query this table first to
    guarantee they use the same master as fit_magnitudes, without
    re-evaluating conditions.
    """

    __tablename__ = "image_master_selection"

    image_id = Column(
        Integer,
        ForeignKey("image.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        doc="The image for which the master was selected.",
    )
    channel = Column(
        String(10),
        primary_key=True,
        doc="The color channel for which the master was selected.",
    )
    master_type_id = Column(
        Integer,
        ForeignKey("master_type.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        doc="The type of master that was selected.",
    )
    master_file_id = Column(
        Integer,
        ForeignKey("master_file.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        doc="The master file that was selected for this image/channel/type.",
    )
