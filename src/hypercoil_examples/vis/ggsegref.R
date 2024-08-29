## Thanks to Sidhant Chopra for providing this example.
## https://sidchop.shinyapps.io/braincoder/

## You need the following package(s) to generate the plot:
library(dplyr)
library(ggseg)
library(ggplot2)
library(ggsegSchaefer)

## Load in atlas data provided by ggseg package
atlas      = as_tibble(schaefer7_400)

## Select atlas region names and hemisphere so that we can add the values
## we want to plot:
region     = atlas$region
hemi       = atlas$hemi
data       = distinct(na.omit(data.frame(region,hemi))) #remove NA and duplicate regions 

## Load in the values you want to plot (one value for each region the the atlas),
## ensuring the values are ordered the same as the region and hemisphere as per the above `data`:
#data$value = sample(length(data$region)) # random data.
data$value = c(1:length(data$region)) # ordered data.
atlas_data = left_join(atlas, data) #merge your values with the atlas data

## Plot atlas:
ggplot() + geom_brain(
                        atlas       = atlas_data,
                        mapping     = aes(fill=value),
                        position    = position_brain(hemi ~ side),
                        hemi        = NULL,
                        color       ='black',
                        size        = 0.5,
                        show.legend = F) +
            theme_void() +
            scale_fill_viridis_c(option='D')
