import numpy as np
import torch


class SpatialAggregation:

    @staticmethod
    def spatial_aggregation_to_coarse_grid(type, scale_factor, y):
        """
        Aggregate a field to a coarser grid.

        Parameters:
        - type: Type of aggregation to use. Options are 'mean' and 'max_pooling'.
        - y: Field to aggregate.
        - scale_factor: Scale factor to use for aggregation.

        Returns:
        - y_coarse: Aggregated field.
        """
        if type == "mean":
            y_coarse = SpatialAggregation.spatial_average_to_coarse_grid(
                y, scale_factor
            )
        elif type == "max":
            y_coarse = SpatialAggregation.max_pooling_to_coarse_grid(y, scale_factor)
        elif type == "ensemble":
            y_coarse = SpatialAggregation.spatial_average_to_coarse_grid(
                y, scale_factor
            )
            if len(y.shape) == 4:
                y_coarse = torch.mean(y_coarse, axis=1)
        else:
            raise ValueError(f"Invalid type of aggregation: {type}")
        return y_coarse

    @staticmethod
    def spatial_average_to_coarse_grid(y, scale_factor):
        """
        Aggregate a field to a coarser grid.

        Parameters:
        - y: Field to aggregate.
        - scale_factor: Scale factor to use for aggregation.

        Returns:
        - y_coarse: Aggregated field.
        """
        y_coarse = torch.nn.functional.avg_pool2d(
            y, scale_factor, scale_factor
        ).squeeze()
        return y_coarse

    @staticmethod
    def max_pooling_to_coarse_grid(y, scale_factor):
        """
        Aggregate a field to a coarser grid using max pooling.

        Parameters:
        - y: Field to aggregate.
        - scale_factor: Scale factor to use for aggregation.

        Returns:
        - y_coarse: Aggregated field.
        """
        y_coarse = torch.nn.functional.max_pool2d(
            y, scale_factor, scale_factor
        ).squeeze()
        return y_coarse
