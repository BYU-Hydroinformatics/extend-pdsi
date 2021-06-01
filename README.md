# GLDAS Data Tools to extend the PDSI dataset

1. Download the GLDAS data (1/4 degree)
1. Download the PDSI dataset
1. Resample to 1/2 degree, 1.25 degree, 2.5 degree 
1. Generate cell assignment tables
    1. From the 


## 1. Download the GLDAS & PDSI datasets
GLDAS dataset
- Temporal extents: 1948-01 to near-real-time
- Monthly averaged
- Spatial extents: (-180, 180) longitude (1440 steps), (-60, 90) latitude (600 steps)
- 1/4 degree lat/lon spatial resolution

PDSI dataset
- Temporal extents: 1850-01 to 2018-12
- Monthly averaged
- Spatial extents: (-180, 180) longitude (144 steps), (-60, 77.5) latitude (55 steps)
- 2.5 degree lat/lon spatial resolution


## 2. Generating the cell assignment tables
1. Use the 2.5 degree PDSI dataset, number each cell in the 2.5 degree lat/lon grid. For each cell:
   - record the x and y index in that 2d array
   - record the x and y values of that cell (that is, the lon and lat of the cell center
   - record the cell number
1. Repeat for the 1.25, .5 and .25 datasets as follows
   - Only query the cells in the higher resolution cells who have the same lat/lon center 
   - Assign them the same cell number as the PDSI grid
   - record the x/y indices, x/y values, and cell number


## 100 training streams
1. Use the script to generate the list of .25 degree cell centroids inside a single 