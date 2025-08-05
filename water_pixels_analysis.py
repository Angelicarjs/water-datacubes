#!/usr/bin/env python3
"""
Standalone script to analyze water pixels by year from existing NDWI data.
This script loads previously calculated NDWI data and creates the yearly analysis.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_pixels_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def plot_water_pixels_by_year(ndwi_data, save_path: str = "water_pixels_by_year.png"):
    """
    Create a comprehensive analysis of water pixels by year from NDWI data.
    
    Parameters:
    -----------
    ndwi_data : xarray.DataArray
        NDWI data with time dimension
    save_path : str
        Path to save the plot
    """
    if ndwi_data is None:
        raise ValueError("NDWI data is None")
    
    logger.info("📊 Creating water pixels by year analysis...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Water Pixels Analysis by Year (2014-2024)', fontsize=16, fontweight='bold')
    
    # Collect data by year
    yearly_data = {}
    monthly_data = {}
    
    for time_val in ndwi_data.time.values:
        date = pd.to_datetime(time_val)
        year = date.year
        month = date.month
        
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
        
        # Store data
        if year not in yearly_data:
            yearly_data[year] = {'pixels': [], 'percentages': [], 'dates': []}
        yearly_data[year]['pixels'].append(water_pixels)
        yearly_data[year]['percentages'].append(water_percentage)
        yearly_data[year]['dates'].append(date)
        
        # Monthly data
        month_key = (year, month)
        if month_key not in monthly_data:
            monthly_data[month_key] = {'pixels': [], 'percentages': []}
        monthly_data[month_key]['pixels'].append(water_pixels)
        monthly_data[month_key]['percentages'].append(water_percentage)
    
    # Plot 1: Water pixels by year (box plot)
    ax1 = axes[0, 0]
    years = sorted(yearly_data.keys())
    water_pixels_by_year = [yearly_data[year]['pixels'] for year in years]
    
    bp1 = ax1.boxplot(water_pixels_by_year, labels=years, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax1.set_title('Water Pixels Distribution by Year')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Water Pixels')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Water percentage by year (box plot)
    ax2 = axes[0, 1]
    water_percentages_by_year = [yearly_data[year]['percentages'] for year in years]
    
    bp2 = ax2.boxplot(water_percentages_by_year, labels=years, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    
    ax2.set_title('Water Percentage Distribution by Year')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Water Percentage (%)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Monthly averages
    ax3 = axes[1, 0]
    months = range(1, 13)
    monthly_avg_pixels = []
    monthly_avg_percentages = []
    
    for month in months:
        month_pixels = []
        month_percentages = []
        for year in years:
            month_key = (year, month)
            if month_key in monthly_data:
                month_pixels.extend(monthly_data[month_key]['pixels'])
                month_percentages.extend(monthly_data[month_key]['percentages'])
        
        if month_pixels:
            monthly_avg_pixels.append(np.mean(month_pixels))
            monthly_avg_percentages.append(np.mean(month_percentages))
        else:
            monthly_avg_pixels.append(0)
            monthly_avg_percentages.append(0)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax3.bar(month_names, monthly_avg_pixels, color='skyblue', alpha=0.8)
    ax3.set_title('Average Water Pixels by Month')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Water Pixels')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Time series trend
    ax4 = axes[1, 1]
    all_dates = []
    all_pixels = []
    all_percentages = []
    
    for year in years:
        all_dates.extend(yearly_data[year]['dates'])
        all_pixels.extend(yearly_data[year]['pixels'])
        all_percentages.extend(yearly_data[year]['percentages'])
    
    # Sort by date
    sorted_data = sorted(zip(all_dates, all_pixels, all_percentages))
    dates, pixels, percentages = zip(*sorted_data)
    
    # Plot time series
    ax4.plot(dates, pixels, 'o-', color='blue', linewidth=2, markersize=4, label='Water Pixels')
    ax4.set_title('Water Pixels Time Series')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Water Pixels')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    
    # Add trend line
    if len(pixels) > 1:
        z = np.polyfit(range(len(pixels)), pixels, 1)
        p = np.poly1d(z)
        ax4.plot(dates, p(range(len(pixels))), "r--", alpha=0.8, label='Trend')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    logger.info("📊 Water Pixels by Year Summary:")
    for year in years:
        year_pixels = yearly_data[year]['pixels']
        year_percentages = yearly_data[year]['percentages']
        logger.info(f"   {year}: {len(year_pixels)} observations")
        logger.info(f"      Avg pixels: {np.mean(year_pixels):.0f} ± {np.std(year_pixels):.0f}")
        logger.info(f"      Avg percentage: {np.mean(year_percentages):.2f}% ± {np.std(year_percentages):.2f}%")
    
    logger.info(f"✅ Water pixels by year analysis saved to {save_path}")
    
    return yearly_data

def load_ndwi_data(file_path: str = "ndwi_data.nc"):
    """
    Load NDWI data from file.
    
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
    
    # Try different file formats
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

def main():
    """
    Main function to run water pixels analysis.
    """
    start_time = time.time()
    logger.info("🚀 Starting Water Pixels Analysis...")
    logger.info("=" * 50)
    
    try:
        # Load NDWI data
        ndwi_data = load_ndwi_data()
        
        # Print data info
        logger.info(f"📊 NDWI Data Info:")
        logger.info(f"   Shape: {ndwi_data.shape}")
        logger.info(f"   Dimensions: {ndwi_data.dims}")
        logger.info(f"   Time range: {ndwi_data.time.min().values} to {ndwi_data.time.max().values}")
        logger.info(f"   Number of time steps: {len(ndwi_data.time)}")
        
        # Create water pixels analysis
        yearly_data = plot_water_pixels_by_year(ndwi_data, save_path="water_pixels_by_year.png")
        
        # Summary
        end_time = time.time()
        logger.info("=" * 50)
        logger.info("🎉 WATER PIXELS ANALYSIS COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
        logger.info("📁 Generated files:")
        logger.info("   📊 water_pixels_by_year.png")
        logger.info("   📄 water_pixels_analysis.log")
        
        return yearly_data
        
    except Exception as e:
        logger.error(f"❌ Error in analysis: {e}")
        logger.error("🔍 Check the log file for detailed error information")
        raise

if __name__ == "__main__":
    print("🚀 Starting Water Pixels Analysis...")
    print("📝 Detailed logs will be saved to 'water_pixels_analysis.log'")
    print("📊 Analysis will be saved to 'water_pixels_by_year.png'")
    print("=" * 60)
    
    try:
        yearly_data = main()
        print("\n" + "=" * 60)
        print("🎉 WATER PIXELS ANALYSIS COMPLETED!")
        print("📊 Check 'water_pixels_by_year.png' for the results")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("🔍 Check the log file for detailed error information")
        sys.exit(1) 