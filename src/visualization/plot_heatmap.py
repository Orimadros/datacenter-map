import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap
from pathlib import Path
import contextily as ctx
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def plot_probability_heatmap(
    tracts_gdf,
    probability_col,
    points_gdf=None,
    output_path=None,
    title="Predicted Probability Heatmap",
    cmap='viridis',
    point_color='blue',
    point_marker='o',
    point_size=20,
    point_edgecolor='black',
    point_linewidth=0.5,
    figsize=(15, 15),
    add_basemap=True,
    basemap_source=ctx.providers.CartoDB.Positron,
    dpi=300,
    custom_vmin=None,
    custom_vmax=None,
    fig=None,
    ax=None,
    show_colorbar=True,
    show_plot=False,
    close_after=True,
    filter_city_code=None,
    city_name=None
):
    """
    Plot a heatmap of predicted probabilities for census tracts with optional overlay of actual points.
    Can optionally filter the plot to a specific city using its code.
    
    Parameters:
    -----------
    tracts_gdf : GeoDataFrame
        GeoDataFrame containing census tract polygons with a column of predicted probabilities.
    probability_col : str
        Name of the column in tracts_gdf containing the predicted probabilities.
    points_gdf : GeoDataFrame, optional
        GeoDataFrame containing point geometries of actual data center locations.
    output_path : str or Path, optional
        Full path including filename to save the output plot. If None, plot won't be saved.
    title : str, default="Predicted Probability Heatmap"
        Title for the plot. Will be modified if city filtering is applied.
    cmap : str or Colormap, default='YlOrRd'
        Colormap for the heatmap.
    point_color : str, default='blue'
        Color for the point markers.
    point_marker : str, default='o'
        Marker style for the points.
    point_size : int, default=20
        Size of the point markers.
    point_edgecolor : str, default='black'
        Edge color for the point markers.
    point_linewidth : float, default=0.5
        Linewidth for the point markers' edge.
    figsize : tuple, default=(15, 15)
        Size of the figure.
    add_basemap : bool, default=True
        Whether to add a basemap underneath the plot (requires internet connection).
    basemap_source : contextily provider, default=ctx.providers.CartoDB.Positron
        Source for the basemap if add_basemap is True.
    dpi : int, default=300
        DPI for the saved figure.
    custom_vmin : float, optional
        Custom minimum value for the colorbar. If None, the minimum probability will be used.
    custom_vmax : float, optional
        Custom maximum value for the colorbar. If None, the maximum probability will be used.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes will be created.
    show_colorbar : bool, default=True
        Whether to show the colorbar on the plot.
    show_plot : bool, default=False
        Whether to display the plot (using plt.show()).
    close_after : bool, default=True
        Whether to close the figure after saving (to free memory).
    filter_city_code : str, optional
        Municipality code (first 7 digits of CD_SETOR) to filter the plot by.
    city_name : str, optional
        Name of the city used for filtering, to be included in the plot title.
    
    Returns:
    --------
    fig, ax : tuple
        The matplotlib figure and axes objects.
    """
    # Input validation
    if not isinstance(tracts_gdf, gpd.GeoDataFrame):
        raise TypeError("tracts_gdf must be a GeoDataFrame")
    
    if probability_col not in tracts_gdf.columns:
        raise ValueError(f"Column '{probability_col}' not found in tracts_gdf")
    
    # Check if tracts_gdf has geometry column for plotting
    if 'geometry' not in tracts_gdf.columns:
        raise ValueError("tracts_gdf must have a 'geometry' column")
    
    # Handle points_gdf if provided
    if points_gdf is not None:
        if not isinstance(points_gdf, gpd.GeoDataFrame):
            raise TypeError("points_gdf must be a GeoDataFrame")
        
        if 'geometry' not in points_gdf.columns:
            raise ValueError("points_gdf must have a 'geometry' column")
    
    # Make copies to avoid modifying the original DataFrames
    tracts_gdf = tracts_gdf.copy()
    if points_gdf is not None:
        points_gdf = points_gdf.copy()
    
    # Filter by city code if provided
    if filter_city_code:
        if 'CD_SETOR' not in tracts_gdf.columns:
            logging.warning("Cannot filter by city: 'CD_SETOR' column not found in tracts_gdf.")
        else:
            logging.info(f"Filtering data for city code: {filter_city_code}...")
            # Ensure CD_SETOR is string and extract city code
            tracts_gdf['CD_MUN'] = tracts_gdf['CD_SETOR'].astype(str).str[:7]
            tracts_gdf = tracts_gdf[tracts_gdf['CD_MUN'] == filter_city_code].copy()
            logging.info(f"Filtered to {len(tracts_gdf)} census tracts for city code {filter_city_code}")

            if len(tracts_gdf) == 0:
                logging.warning(f"No tracts found for city code {filter_city_code}. Plot will be empty.")
                # Optionally return early or proceed with empty plot
                # return fig, ax # Or create empty fig/ax if they don't exist

            # Update title
            city_display_name = city_name if city_name else f"City {filter_city_code}"
            title = f"{title} - {city_display_name}"

            # Filter points within the city boundary
            if points_gdf is not None and len(tracts_gdf) > 0:
                # Ensure CRS match before spatial operations
                if points_gdf.crs != tracts_gdf.crs:
                    logging.info(f"Aligning CRS for points filtering: {points_gdf.crs} -> {tracts_gdf.crs}")
                    try:
                        points_gdf = points_gdf.to_crs(tracts_gdf.crs)
                    except Exception as e:
                        logging.warning(f"Could not align CRS for points: {e}. Skipping points filtering.")
                        points_gdf = None # Or handle differently
                
                if points_gdf is not None:
                    city_boundary = tracts_gdf.geometry.unary_union
                    points_gdf = points_gdf[points_gdf.geometry.within(city_boundary)].copy()
                    logging.info(f"Filtered to {len(points_gdf)} points within city boundaries.")

    # For basemap compatibility, reproject to Web Mercator (EPSG:3857)
    if add_basemap:
        web_mercator_crs = "EPSG:3857"
        original_crs = tracts_gdf.crs
        if tracts_gdf.empty:
            add_basemap = False # Can't reproject empty GDF
            logging.warning("Cannot add basemap as tracts GeoDataFrame is empty after filtering.")
        else:
            try:
                logging.info(f"Reprojecting data to {web_mercator_crs} for basemap compatibility")
                tracts_gdf = tracts_gdf.to_crs(web_mercator_crs)
                if points_gdf is not None and not points_gdf.empty:
                    points_gdf = points_gdf.to_crs(web_mercator_crs)
            except Exception as e:
                logging.warning(f"Could not reproject data: {e}")
                logging.warning("Continuing with original CRS, basemap may not align properly.")
                add_basemap = False  # Disable basemap if reprojection fails
    
    # Set up figure and axes
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Check if tracts_gdf is empty before plotting
    if tracts_gdf.empty:
        logging.warning("No data to plot after filtering.")
    else:
        # Plot the heatmap
        vmin = custom_vmin if custom_vmin is not None else tracts_gdf[probability_col].min()
        vmax = custom_vmax if custom_vmax is not None else tracts_gdf[probability_col].max()
        
        tracts_gdf.plot(
            column=probability_col,
            cmap=cmap,
            linewidth=0.2,
            ax=ax,
            edgecolor='0.5',
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
        )
        
        # Add colorbar
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Probability')
        
        # Overlay actual data center points if provided and not empty
        if points_gdf is not None and not points_gdf.empty:
            points_gdf.plot(
                ax=ax,
                color=point_color,
                marker=point_marker,
                markersize=point_size,
                edgecolor=point_edgecolor,
                linewidth=point_linewidth,
                zorder=5,  # Ensure points are above the heatmap
                label="Data Centers" # Add label for legend
            )
            ax.legend() # Show legend if points are plotted
        
        # Add basemap if requested
        if add_basemap:
            try:
                ctx.add_basemap(
                    ax,
                    source=basemap_source,
                    attribution_size=8,
                )
            except Exception as e:
                logging.warning(f"Could not add basemap: {e}")
                logging.warning("Continuing without basemap. Check internet connection or basemap provider.")
    
    # Set plot title and remove axis labels
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    
    # Save the plot if output_path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logging.info(f"Plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save plot: {e}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    # Close figure to free memory if requested
    if close_after and not show_plot:
        plt.close(fig)
    
    return fig, ax

def plot_multiple_models_comparison(
    tracts_gdf,
    model_results_dict,
    points_gdf=None,
    output_path=None,
    main_title="Model Comparison",
    cmap='YlOrRd',
    figsize=(20, 15),
    dpi=300,
):
    """
    Create a multi-panel figure comparing probability heatmaps from different models.
    
    Parameters:
    -----------
    tracts_gdf : GeoDataFrame
        Base GeoDataFrame containing census tract polygons (without probabilities).
    model_results_dict : dict
        Dictionary with format {model_name: probability_column_name} where:
        - model_name (str): Name/label for the model
        - probability_column_name (str): Column name in tracts_gdf with probabilities from this model
    points_gdf : GeoDataFrame, optional
        GeoDataFrame containing point geometries of actual data center locations.
    output_path : str or Path, optional
        Full path including filename to save the output plot. If None, plot won't be saved.
    main_title : str, default="Model Comparison"
        Main title for the overall figure.
    cmap : str or Colormap, default='YlOrRd'
        Colormap for the heatmap.
    figsize : tuple, default=(20, 15)
        Size of the figure.
    dpi : int, default=300
        DPI for the saved figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    # Determine grid layout based on number of models
    n_models = len(model_results_dict)
    if n_models <= 1:
        logging.warning("Multiple model comparison requires at least 2 models")
        if n_models == 1:
            model_name = list(model_results_dict.keys())[0]
            prob_col = model_results_dict[model_name]
            return plot_probability_heatmap(
                tracts_gdf=tracts_gdf,
                probability_col=prob_col,
                points_gdf=points_gdf,
                output_path=output_path,
                title=model_name,
                cmap=cmap,
                figsize=figsize,
                dpi=dpi
            )
        return None, None
    
    # Calculate grid dimensions
    if n_models <= 2:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 4:
        n_cols = 2
        n_rows = (n_models + 1) // 2
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Calculate global min/max for consistent color scaling
    prob_columns = list(model_results_dict.values())
    global_vmin = min(tracts_gdf[col].min() for col in prob_columns if col in tracts_gdf.columns)
    global_vmax = max(tracts_gdf[col].max() for col in prob_columns if col in tracts_gdf.columns)
    
    # Plot each model
    for i, (model_name, prob_col) in enumerate(model_results_dict.items()):
        if i < len(axes):
            if prob_col not in tracts_gdf.columns:
                logging.warning(f"Column '{prob_col}' not found in tracts_gdf. Skipping {model_name}")
                continue
                
            # Plot on the corresponding subplot
            plot_probability_heatmap(
                tracts_gdf=tracts_gdf,
                probability_col=prob_col,
                points_gdf=points_gdf,
                title=model_name,
                cmap=cmap,
                fig=fig,
                ax=axes[i],
                show_colorbar=True,
                close_after=False,
                custom_vmin=global_vmin,
                custom_vmax=global_vmax,
                add_basemap=True,
            )
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Add overall title
    plt.suptitle(main_title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for overall title
    
    # Save the plot if output_path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logging.info(f"Comparison plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save comparison plot: {e}")
    
    return fig, axes

def example_usage(show_examples=False):
    """Sample usage examples for documentation purposes."""
    if show_examples:
        # Import necessary libraries for this example
        import random
        
        # 1. Create sample data
        # Load census tract data (assumes file exists)
        try:
            project_root = Path(__file__).parent.parent.parent
            tracts_path = project_root / "data/model_input/brasil/sao_paulo_census_tracts_full.geojson"
            tracts_gdf = gpd.read_file(tracts_path)
            
            # 2. Generate random probabilities for demonstration
            tracts_gdf['xgboost_prob'] = np.random.random(len(tracts_gdf)) * 0.1
            tracts_gdf['nn_prob'] = np.random.random(len(tracts_gdf)) * 0.1
            
            # Set higher probabilities for some tracts to simulate model predictions
            high_prob_indices = np.random.choice(len(tracts_gdf), 100, replace=False)
            tracts_gdf.loc[high_prob_indices, 'xgboost_prob'] = 0.3 + np.random.random(len(high_prob_indices)) * 0.7
            tracts_gdf.loc[high_prob_indices, 'nn_prob'] = 0.2 + np.random.random(len(high_prob_indices)) * 0.8
            
            # Get actual data centers for demonstration
            # This assumes there's a has_data_center column in the original data
            if 'has_data_center' in tracts_gdf.columns:
                dc_tracts = tracts_gdf[tracts_gdf['has_data_center'] == 1].copy()
                # Create points by using centroids of data center tracts
                points_gdf = dc_tracts.copy()
                points_gdf['geometry'] = points_gdf.geometry.centroid
            else:
                # Alternatively, create random points
                logging.warning("No 'has_data_center' column found. Generating random point locations.")
                points_gdf = None
            
            # 3. Create output directory
            plots_dir = project_root / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # 4. Single model plot
            single_plot_path = plots_dir / "example_xgboost_heatmap.png"
            plot_probability_heatmap(
                tracts_gdf=tracts_gdf,
                probability_col='xgboost_prob',
                points_gdf=points_gdf,
                output_path=single_plot_path,
                title="XGBoost Predicted Probabilities",
                show_plot=True
            )
            
            # 5. Comparison plot
            comparison_plot_path = plots_dir / "example_model_comparison.png"
            model_dict = {
                "XGBoost Model": "xgboost_prob",
                "Neural Network Model": "nn_prob"
            }
            plot_multiple_models_comparison(
                tracts_gdf=tracts_gdf,
                model_results_dict=model_dict,
                points_gdf=points_gdf,
                output_path=comparison_plot_path,
                main_title="Model Comparison: XGBoost vs Neural Network",
                show_plot=True
            )
            
            logging.info("Example visualizations completed!")
            
        except Exception as e:
            logging.error(f"Example visualization failed: {e}")
    else:
        # Code example only - not executed
        logging.info("Example code (not executed):")
        logging.info("""
        # Single model heatmap
        plot_probability_heatmap(
            tracts_gdf=gdf_with_predictions,
            probability_col='xgboost_probabilities',
            points_gdf=actual_datacenters_gdf,
            output_path='path/to/save/heatmap.png',
            title="XGBoost Predicted Probabilities",
        )
        
        # Multiple models comparison
        model_dict = {
            "XGBoost": "xgboost_probabilities",
            "Neural Network": "nn_probabilities" 
        }
        plot_multiple_models_comparison(
            tracts_gdf=gdf_with_predictions,
            model_results_dict=model_dict,
            points_gdf=actual_datacenters_gdf,
            output_path='path/to/save/comparison.png',
            main_title="Model Comparison: XGBoost vs Neural Network",
        )
        """)

if __name__ == "__main__":
    # Show documentation example without running actual code
    example_usage(show_examples=False)
    
    logging.info("This script is designed to be imported and used from other scripts.")
    logging.info("Run with show_examples=True to execute example visualizations, which requires data files.") 