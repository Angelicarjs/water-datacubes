#!/usr/bin/env python3
"""
Script to analyze correlation between cloud cover and water pixels classification.
This helps understand if cloud cover affects water detection accuracy.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from datetime import datetime
import time
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloud_water_correlation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_satellite_data(file_path: str = "satellite_data.nc"):
    """
    Load satellite data including cloud cover information.
    
    Parameters:
    -----------
    file_path : str
        Path to the satellite data file
        
    Returns:
    --------
    xarray.Dataset
        Loaded satellite data with cloud information
    """
    logger.info(f"📂 Loading satellite data from {file_path}...")
    
    try:
        # Try NetCDF format
        data = xr.open_dataset(file_path)
        logger.info("✅ Satellite data loaded successfully (NetCDF format)")
        return data
    except Exception as e1:
        logger.warning(f"⚠️  NetCDF load failed: {e1}")
        
        try:
            # Try HDF5 format
            hdf5_path = file_path.replace('.nc', '.h5')
            data = xr.open_dataset(hdf5_path)
            logger.info("✅ Satellite data loaded successfully (HDF5 format)")
            return data
        except Exception as e2:
            logger.warning(f"⚠️  HDF5 load failed: {e2}")
            
            try:
                # Try pickle format
                pickle_path = file_path.replace('.nc', '.pkl')
                import pickle
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info("✅ Satellite data loaded successfully (Pickle format)")
                return data
            except Exception as e3:
                logger.error(f"❌ All load attempts failed:")
                logger.error(f"   NetCDF: {e1}")
                logger.error(f"   HDF5: {e2}")
                logger.error(f"   Pickle: {e3}")
                raise ValueError("Could not load satellite data from any supported format")

def load_ndwi_data(file_path: str = "ndwi_data.nc"):
    """
    Load NDWI data.
    
    Parameters:
    -----------
    file_path : str
        Path to the NDWI data file
        
    Returns:
    --------
    xarray.DataArray
        Loaded NDWI data
    """
    logger.info(f"📂 Loading NDWI data from {file_path}...")
    
    try:
        # Try NetCDF format
        data = xr.open_dataarray(file_path)
        logger.info("✅ NDWI data loaded successfully (NetCDF format)")
        return data
    except Exception as e1:
        logger.warning(f"⚠️  NetCDF load failed: {e1}")
        
        try:
            # Try HDF5 format
            hdf5_path = file_path.replace('.nc', '.h5')
            data = xr.open_dataarray(hdf5_path)
            logger.info("✅ NDWI data loaded successfully (HDF5 format)")
            return data
        except Exception as e2:
            logger.warning(f"⚠️  HDF5 load failed: {e2}")
            
            try:
                # Try pickle format
                pickle_path = file_path.replace('.nc', '.pkl')
                import pickle
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info("✅ NDWI data loaded successfully (Pickle format)")
                return data
            except Exception as e3:
                logger.error(f"❌ All load attempts failed:")
                logger.error(f"   NetCDF: {e1}")
                logger.error(f"   HDF5: {e2}")
                logger.error(f"   Pickle: {e3}")
                raise ValueError("Could not load NDWI data from any supported format")

def extract_cloud_water_data(satellite_data, ndwi_data):
    """
    Extract cloud cover and water pixels data for correlation analysis.
    
    Parameters:
    -----------
    satellite_data : xarray.Dataset
        Satellite data with cloud information
    ndwi_data : xarray.DataArray
        NDWI data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cloud cover and water pixels data
    """
    logger.info("📊 Extracting cloud cover and water pixels data...")
    
    analysis_data = []
    
    for time_val in ndwi_data.time.values:
        date = pd.to_datetime(time_val)
        
        # Get NDWI data for this time step
        ndwi_slice = ndwi_data.sel(time=time_val)
        
        # Ensure the slice is 2D for analysis
        if len(ndwi_slice.dims) > 2:
            extra_dims = [dim for dim in ndwi_slice.dims if dim not in ['y', 'x', 'lat', 'lon']]
            for dim in extra_dims:
                ndwi_slice = ndwi_slice.isel({dim: 0})
        
        # Convert to numpy array and ensure it's 2D
        ndwi_array = ndwi_slice.values
        if ndwi_array.ndim > 2:
            ndwi_array = ndwi_array[0] if ndwi_array.shape[0] == 1 else ndwi_array.squeeze()
        
        # Count water pixels
        water_mask = ndwi_array > 0.3
        water_pixels = water_mask.sum()
        total_pixels = ndwi_array.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Try to get cloud cover from satellite data
        cloud_cover = None
        try:
            # Try different possible cloud cover variables
            cloud_vars = ['cloud_cover', 'eo:cloud_cover', 'eo:cloudCover', 'cloudcover', 'cloudCover']
            for var in cloud_vars:
                if var in satellite_data.data_vars:
                    cloud_slice = satellite_data[var].sel(time=time_val)
                    cloud_cover = float(cloud_slice.values)
                    break
            
            # If no cloud cover variable found, try to estimate from band data
            if cloud_cover is None:
                logger.warning(f"⚠️  No cloud cover data found for {date}, estimating from band data...")
                # Simple cloud detection using brightness threshold
                if 'green' in satellite_data.data_vars:
                    green_slice = satellite_data['green'].sel(time=time_val)
                    green_array = green_slice.values
                    if green_array.ndim > 2:
                        green_array = green_array.squeeze()
                    
                    # Estimate cloud cover from brightness (simple method)
                    brightness = np.mean(green_array)
                    # Normalize brightness to 0-100% (this is a rough estimate)
                    cloud_cover = min(100, max(0, (brightness - 0.1) * 1000))
                else:
                    cloud_cover = 0  # Default if no data available
                    
        except Exception as e:
            logger.warning(f"⚠️  Could not extract cloud cover for {date}: {e}")
            cloud_cover = 0
        
        # Store data
        analysis_data.append({
            'date': date,
            'year': date.year,
            'month': date.month,
            'water_pixels': water_pixels,
            'water_percentage': water_percentage,
            'cloud_cover': cloud_cover,
            'total_pixels': total_pixels
        })
    
    df = pd.DataFrame(analysis_data)
    logger.info(f"✅ Extracted data for {len(df)} time steps")
    
    return df

def analyze_cloud_water_correlation(df, save_path: str = "cloud_water_correlation.png"):
    """
    Analyze correlation between cloud cover and water pixels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with cloud cover and water pixels data
    save_path : str
        Path to save the analysis plots
    """
    logger.info("📊 Analyzing cloud-water correlation...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cloud Cover vs Water Pixels Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Filter out invalid data
    valid_data = df[(df['cloud_cover'] >= 0) & (df['water_pixels'] >= 0)].copy()
    
    if len(valid_data) == 0:
        logger.error("❌ No valid data for correlation analysis")
        return
    
    # Plot 1: Scatter plot - Cloud cover vs Water pixels
    ax1 = axes[0, 0]
    ax1.scatter(valid_data['cloud_cover'], valid_data['water_pixels'], 
               alpha=0.6, color='blue', s=50)
    
    # Add trend line
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['cloud_cover'], valid_data['water_pixels'], 1)
        p = np.poly1d(z)
        ax1.plot(valid_data['cloud_cover'], p(valid_data['cloud_cover']), 
                "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Cloud Cover (%)')
    ax1.set_ylabel('Water Pixels')
    ax1.set_title('Cloud Cover vs Water Pixels')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot - Cloud cover vs Water percentage
    ax2 = axes[0, 1]
    ax2.scatter(valid_data['cloud_cover'], valid_data['water_percentage'], 
               alpha=0.6, color='green', s=50)
    
    # Add trend line
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['cloud_cover'], valid_data['water_percentage'], 1)
        p = np.poly1d(z)
        ax2.plot(valid_data['cloud_cover'], p(valid_data['cloud_cover']), 
                "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Cloud Cover (%)')
    ax2.set_ylabel('Water Percentage (%)')
    ax2.set_title('Cloud Cover vs Water Percentage')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation heatmap
    ax3 = axes[0, 2]
    correlation_matrix = valid_data[['cloud_cover', 'water_pixels', 'water_percentage']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax3, cbar_kws={'label': 'Correlation Coefficient'})
    ax3.set_title('Correlation Matrix')
    
    # Plot 4: Cloud cover distribution
    ax4 = axes[1, 0]
    ax4.hist(valid_data['cloud_cover'], bins=20, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True)
    ax4.set_xlabel('Cloud Cover (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('Cloud Cover Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Water pixels by cloud cover categories
    ax5 = axes[1, 1]
    # Create cloud cover categories
    valid_data['cloud_category'] = pd.cut(valid_data['cloud_cover'], 
                                         bins=[0, 10, 25, 50, 100], 
                                         labels=['Clear (0-10%)', 'Low (10-25%)', 'Medium (25-50%)', 'High (50-100%)'])
    
    cloud_water_stats = valid_data.groupby('cloud_category')['water_pixels'].agg(['mean', 'std', 'count']).dropna()
    
    if len(cloud_water_stats) > 0:
        x_pos = range(len(cloud_water_stats))
        ax5.bar(x_pos, cloud_water_stats['mean'], yerr=cloud_water_stats['std'], 
               capsize=5, alpha=0.7, color='skyblue')
        ax5.set_xlabel('Cloud Cover Category')
        ax5.set_ylabel('Average Water Pixels')
        ax5.set_title('Water Pixels by Cloud Cover Category')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(cloud_water_stats.index, rotation=45)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Time series of cloud cover and water pixels
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    
    # Sort by date
    valid_data_sorted = valid_data.sort_values('date')
    
    # Plot cloud cover on primary y-axis
    line1 = ax6.plot(valid_data_sorted['date'], valid_data_sorted['cloud_cover'], 
                     'o-', color='gray', linewidth=2, markersize=4, label='Cloud Cover')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Cloud Cover (%)', color='gray')
    ax6.tick_params(axis='y', labelcolor='gray')
    ax6.grid(True, alpha=0.3)
    
    # Plot water pixels on secondary y-axis
    line2 = ax6_twin.plot(valid_data_sorted['date'], valid_data_sorted['water_pixels'], 
                          's-', color='blue', linewidth=2, markersize=4, label='Water Pixels')
    ax6_twin.set_ylabel('Water Pixels', color='blue')
    ax6_twin.tick_params(axis='y', labelcolor='blue')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left')
    
    ax6.set_title('Cloud Cover and Water Pixels Over Time')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation statistics
    correlation_stats = calculate_correlation_statistics(valid_data)
    
    # Print summary
    logger.info("📊 Cloud-Water Correlation Analysis Summary:")
    logger.info(f"   Total observations: {len(valid_data)}")
    logger.info(f"   Cloud cover range: {valid_data['cloud_cover'].min():.1f}% - {valid_data['cloud_cover'].max():.1f}%")
    logger.info(f"   Water pixels range: {valid_data['water_pixels'].min():.0f} - {valid_data['water_pixels'].max():.0f}")
    logger.info(f"   Correlation coefficient (cloud vs water pixels): {correlation_stats['pearson_r']:.3f}")
    logger.info(f"   P-value: {correlation_stats['pearson_p']:.3f}")
    logger.info(f"   R-squared: {correlation_stats['r_squared']:.3f}")
    
    logger.info(f"✅ Cloud-water correlation analysis saved to {save_path}")
    
    return correlation_stats

def calculate_correlation_statistics(df):
    """
    Calculate correlation statistics between cloud cover and water pixels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with cloud cover and water pixels data
        
    Returns:
    --------
    dict
        Dictionary with correlation statistics
    """
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(df['cloud_cover'], df['water_pixels'])
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(df['cloud_cover'], df['water_pixels'])
    
    # Linear regression
    X = df['cloud_cover'].values.reshape(-1, 1)
    y = df['water_pixels'].values
    reg = LinearRegression().fit(X, y)
    r_squared = r2_score(y, reg.predict(X))
    
    # Calculate confidence intervals
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'n_observations': len(df)
    }

def create_cloud_masking_analysis(df, save_path: str = "cloud_masking_analysis.png"):
    """
    Create analysis of how cloud masking affects water detection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with cloud cover and water pixels data
    save_path : str
        Path to save the analysis plots
    """
    logger.info("📊 Creating cloud masking analysis...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cloud Masking Impact on Water Detection', fontsize=16, fontweight='bold')
    
    # Filter valid data
    valid_data = df[(df['cloud_cover'] >= 0) & (df['water_pixels'] >= 0)].copy()
    
    if len(valid_data) == 0:
        logger.error("❌ No valid data for cloud masking analysis")
        return
    
    # Plot 1: Water pixels vs cloud cover with different thresholds
    ax1 = axes[0, 0]
    cloud_thresholds = [0, 10, 25, 50, 75, 90]
    colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    
    for i, threshold in enumerate(cloud_thresholds):
        if threshold == 0:
            mask = valid_data['cloud_cover'] <= threshold
            label = f'Cloud cover ≤ {threshold}%'
        else:
            mask = valid_data['cloud_cover'] > cloud_thresholds[i-1]
            label = f'{cloud_thresholds[i-1]}% < Cloud cover ≤ {threshold}%'
        
        subset = valid_data[mask]
        if len(subset) > 0:
            ax1.scatter(subset['cloud_cover'], subset['water_pixels'], 
                       alpha=0.7, color=colors[i], s=50, label=label)
    
    ax1.set_xlabel('Cloud Cover (%)')
    ax1.set_ylabel('Water Pixels')
    ax1.set_title('Water Pixels by Cloud Cover Thresholds')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average water pixels by cloud cover range
    ax2 = axes[0, 1]
    valid_data['cloud_range'] = pd.cut(valid_data['cloud_cover'], 
                                      bins=[0, 10, 25, 50, 75, 100], 
                                      labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-100%'])
    
    cloud_stats = valid_data.groupby('cloud_range').agg({
        'water_pixels': ['mean', 'std', 'count'],
        'water_percentage': ['mean', 'std']
    }).dropna()
    
    if len(cloud_stats) > 0:
        x_pos = range(len(cloud_stats))
        means = cloud_stats[('water_pixels', 'mean')]
        stds = cloud_stats[('water_pixels', 'std')]
        
        ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='lightblue')
        ax2.set_xlabel('Cloud Cover Range')
        ax2.set_ylabel('Average Water Pixels')
        ax2.set_title('Average Water Pixels by Cloud Cover Range')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cloud_stats.index, rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cloud cover impact on water detection accuracy
    ax3 = axes[1, 0]
    # Calculate coefficient of variation (CV) as a measure of variability
    cloud_stats_cv = valid_data.groupby('cloud_range')['water_pixels'].agg(['mean', 'std']).dropna()
    cloud_stats_cv['cv'] = cloud_stats_cv['std'] / cloud_stats_cv['mean']
    
    if len(cloud_stats_cv) > 0:
        x_pos = range(len(cloud_stats_cv))
        ax3.bar(x_pos, cloud_stats_cv['cv'], alpha=0.7, color='orange')
        ax3.set_xlabel('Cloud Cover Range')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Water Detection Variability by Cloud Cover')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(cloud_stats_cv.index, rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recommended cloud cover thresholds
    ax4 = axes[1, 1]
    # Calculate optimal cloud cover threshold based on water detection stability
    thresholds = np.arange(0, 100, 5)
    stability_scores = []
    
    for threshold in thresholds:
        low_cloud_data = valid_data[valid_data['cloud_cover'] <= threshold]
        if len(low_cloud_data) > 1:
            # Calculate stability as inverse of coefficient of variation
            mean_water = low_cloud_data['water_pixels'].mean()
            std_water = low_cloud_data['water_pixels'].std()
            if mean_water > 0:
                cv = std_water / mean_water
                stability = 1 / (1 + cv)  # Higher is better
            else:
                stability = 0
        else:
            stability = 0
        stability_scores.append(stability)
    
    ax4.plot(thresholds, stability_scores, 'o-', color='purple', linewidth=2, markersize=6)
    ax4.set_xlabel('Cloud Cover Threshold (%)')
    ax4.set_ylabel('Water Detection Stability Score')
    ax4.set_title('Optimal Cloud Cover Threshold')
    ax4.grid(True, alpha=0.3)
    
    # Find optimal threshold
    optimal_threshold = thresholds[np.argmax(stability_scores)]
    ax4.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal: {optimal_threshold}%')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Cloud masking analysis saved to {save_path}")
    logger.info(f"📊 Recommended cloud cover threshold: {optimal_threshold}%")
    
    return optimal_threshold

def main():
    """
    Main function to run cloud-water correlation analysis.
    """
    start_time = time.time()
    logger.info("🚀 Starting Cloud-Water Correlation Analysis...")
    logger.info("=" * 50)
    
    try:
        # Load data
        satellite_data = load_satellite_data()
        ndwi_data = load_ndwi_data()
        
        # Print data info
        logger.info(f"📊 Satellite Data Info:")
        logger.info(f"   Variables: {list(satellite_data.data_vars)}")
        logger.info(f"   Time range: {satellite_data.time.min().values} to {satellite_data.time.max().values}")
        
        logger.info(f"📊 NDWI Data Info:")
        logger.info(f"   Shape: {ndwi_data.shape}")
        logger.info(f"   Dimensions: {ndwi_data.dims}")
        logger.info(f"   Time range: {ndwi_data.time.min().values} to {ndwi_data.time.max().values}")
        
        # Extract correlation data
        df = extract_cloud_water_data(satellite_data, ndwi_data)
        
        # Analyze correlation
        correlation_stats = analyze_cloud_water_correlation(df, save_path="cloud_water_correlation.png")
        
        # Create cloud masking analysis
        optimal_threshold = create_cloud_masking_analysis(df, save_path="cloud_masking_analysis.png")
        
        # Summary
        end_time = time.time()
        logger.info("=" * 50)
        logger.info("🎉 CLOUD-WATER CORRELATION ANALYSIS COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        logger.info("📁 Generated files:")
        logger.info("   📊 cloud_water_correlation.png")
        logger.info("   📊 cloud_masking_analysis.png")
        logger.info("   📄 cloud_water_correlation.log")
        logger.info(f"📊 Key findings:")
        logger.info(f"   Correlation coefficient: {correlation_stats['pearson_r']:.3f}")
        logger.info(f"   P-value: {correlation_stats['pearson_p']:.3f}")
        logger.info(f"   R-squared: {correlation_stats['r_squared']:.3f}")
        logger.info(f"   Recommended cloud threshold: {optimal_threshold}%")
        
        return df, correlation_stats, optimal_threshold
        
    except Exception as e:
        logger.error(f"❌ Error in analysis: {e}")
        logger.error("🔍 Check the log file for detailed error information")
        raise

if __name__ == "__main__":
    print("🚀 Starting Cloud-Water Correlation Analysis...")
    print("📝 Detailed logs will be saved to 'cloud_water_correlation.log'")
    print("📊 Analysis will be saved to correlation plots")
    print("=" * 60)
    
    try:
        df, correlation_stats, optimal_threshold = main()
        print("\n" + "=" * 60)
        print("🎉 CLOUD-WATER CORRELATION ANALYSIS COMPLETED!")
        print(f"📊 Correlation coefficient: {correlation_stats['pearson_r']:.3f}")
        print(f"📊 Recommended cloud threshold: {optimal_threshold}%")
        print("📊 Check the generated plots for detailed analysis")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("🔍 Check the log file for detailed error information")
        sys.exit(1) 