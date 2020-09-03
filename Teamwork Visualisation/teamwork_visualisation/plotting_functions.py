import numpy as np
import pandas as pd
#import graphviz
import seaborn as sns
import re
from IPython.core.display import display, SVG
import teamwork_visualisation.energy_engagement as ee
import math
import os
import warnings

colour_palette = sns.color_palette(
    ['#3b9ab2', '#409cb4', '#459fb5', '#4aa1b7', '#4fa3b8', '#54a6ba', '#59a8bb', '#5eabbd', '#63adbe', '#68afc0', '#6db2c2', '#72b4c3', '#77b6c5', '#7fb8bc', '#88baaf', '#92bca2', '#9bbd96', '#a5bf89', '#aec17c', '#b7c370', '#c1c463', '#cac656', '#d4c84a', '#ddc93d', '#e6cb30', '#ebcb28', '#eac825', '#e9c622', '#e8c41f', '#e8c11c', '#e7bf18', '#e6bd15', '#e5ba12', '#e5b80f', '#e4b60c', '#e3b309', '#e3b105', '#e2ac04', '#e0a207', '#df970a', '#de8d0d', '#dd830f', '#dc7812', '#da6e15', '#d96318', '#d8591b', '#d74e1e', '#d54421', '#d43924', '#d32f27']
)

def plot_raw_volume(raw_volume, key_points=None, use_key_point_labels=None):
    
    raw_volume = raw_volume.resample("5S").mean()
    
    ## Find an appropriate resample rate so that results display correctly - basically there should not be 
    ## too many cells in the heatmap
    ## We find the nice, round period (ranging from 5 seconds to 5 minutes) that gets us closest to 200 observations 
    period_multiples = np.array([1,2,3,6,12,24,36,48,60])     ## Remember that it's already at a rate of 5 seconds 
    resampling_rate = str(5 * period_multiples[np.argmin(np.abs(200 - (len(raw_volume) / period_multiples)))]) + "S"
    
    volume_resampled = raw_volume.resample(resampling_rate).mean()
    
    ## Create the plot and edit it a bit to improve labels, etc.
    
    plot = sns.heatmap(volume_resampled.T, cmap=colour_palette)

    plot.set_xticklabels(volume_resampled.index[plot.get_xticks().astype(int)].strftime('%H:%M:%S'))
    plot.set_xlabel("Time (hours:minutes:seconds)")
    plot.set_ylabel("")
    plot.get_figure().set_size_inches(12,+ (np.log(len(volume_resampled.columns))/ np.log(1.6)))
    
    ## We might also want to plot some user-provided key points (for example if someone recorded the times of important events)
    if key_points is not None:
        
        axes1 = plot.axes
        axes2 = axes1.twiny()

        axes2.set_xlim(volume_resampled.index.astype(int).min(), volume_resampled.index.astype(int).max())
        
        if key_points.index.astype(int).min() < raw_volume.index.astype(int).min():
            raise ValueError("Earliest key point is before the first time displayed in the heatmap")
            
        if key_points.index.astype(int).max() > raw_volume.index.astype(int).max():
            raise ValueError("Latest key point is after the final time displayed in the heatmap")
            
        axes2.set_xticks(key_points.index.astype(int))

        if use_key_point_labels:
            axes2.set_xticklabels(key_points.tolist())
        else:
            axes2.set_xticklabels(key_points.index.strftime('%H:%M:%S'))        

        axes2.get_figure().autofmt_xdate(rotation=90)
        
    return plot
    


def plot_dynamic_complexity(results, critical_instabilities, output_folder, key_points=None, use_key_point_labels=True):
    
    ## Find an appropriate resample rate so that results display correctly - basically there should not be 
    ## too many cells in the heatmap
    ## We find the nice, round period (ranging from 5 seconds to 5 minutes) that gets us closest to 200 observations 
    period_multiples = np.array([1,2,3,6,12,24,36,48,60])     ## Remember that it's already at a rate of 5 seconds 
    resampling_rate = str(5 * period_multiples[np.argmin(np.abs(200 - (len(results) / period_multiples)))]) + "S"
    
    results_resampled = results.resample(resampling_rate).mean()
    
    critical_instabilities_resampled = critical_instabilities.resample(resampling_rate).max()
    
    ## Create the plot and edit it a bit to improve labels, etc.
    
    plot = sns.heatmap(pd.concat([results_resampled.T / results_resampled.max().max(), 
                           results_resampled.mean(axis=1).to_frame(name="Mean Dynamic Complexity").T / results_resampled.mean(axis=1).max(),
                           critical_instabilities_resampled.to_frame(name="Critical Instabilities").T], axis=0),
                      cmap=colour_palette)

    plot.set_xticklabels(results_resampled.index[plot.get_xticks().astype(int)].strftime('%H:%M:%S'))
    plot.set_xlabel("Time (hours:minutes:seconds)")
    plot.get_figure().set_size_inches(12,0.5 + (np.log(len(results_resampled.columns))/ np.log(1.6)))

    ## We might also want to plot some user-provided key points (for example if someone recorded the times of important events)
    if key_points is not None:
        
        axes1 = plot.axes
        axes2 = axes1.twiny()

        axes2.set_xlim(results_resampled.index.astype(int).min(), results_resampled.index.astype(int).max())
        
        if key_points.index.astype(int).min() < results.index.astype(int).min():
            raise ValueError("Earliest key point is before the first time displayed in the heatmap")
            
        if key_points.index.astype(int).max() > results.index.astype(int).max():
            raise ValueError("Latest key point is after the final time displayed in the heatmap")
            
        axes2.set_xticks(key_points.index.astype(int))

        if use_key_point_labels:
            axes2.set_xticklabels(key_points.tolist())
        else:
            axes2.set_xticklabels(key_points.index.strftime('%H:%M:%S'))        

        axes2.get_figure().autofmt_xdate(rotation=90)
    
    plot.get_figure().savefig(os.path.join(output_folder, "dynamic_complexity.png"), bbox_inches="tight")
    
    return plot


def rotate_point(x, y, angle):
    
    angle = angle * (math.pi / 180); ## Convert to radians
    rotatedX = x * math.cos(angle) - y * math.sin(angle)
    rotatedY = x * math.sin(angle) + y * math.cos(angle)
    
    return rotatedX, rotatedY


def get_node_positions(start_x, start_y, number_nodes):
    
    base_angle = 360 / number_nodes
    
    return [rotate_point(start_x, start_y, base_angle * num) for num in range(number_nodes)]


def get_node_data(per_second_speech):
    
    energy = ee.get_energy(per_second_speech)

    return pd.concat([energy, 
                      pd.DataFrame(
                          get_node_positions(0, 115, len(energy)), 
                          index=energy.index, 
                          columns=["x", "y"]
                      )], 
                     axis=1)


def get_edge_data(per_second_speech):
    
    engagement = ee.get_engagement(per_second_speech)
    
    edge_data = engagement.merge(get_node_data(per_second_speech), left_on=engagement.index.get_level_values(0), right_index=True).drop(columns=["key_0"])
    
    edge_data = edge_data.merge(get_node_data(per_second_speech), left_on=edge_data.index.get_level_values(1), right_index=True, suffixes=("_1", "_2")).drop(columns=["key_0"])
    
    return edge_data


node_template = """<g id="{0}" class="node"><title>{1}</title>
<ellipse fill="#EDB81D" stroke="black" stroke-width="0" cx="{2:.2f}" cy="{3:.2f}" rx="{4:.2f}" ry="{4:.2f}"/>
<text text-anchor="middle" x="{5:.2f}" y="{6:.2f}" font-family="DejaVu Sans,sans-serif" font-size="{7}">{8}</text>
</g>"""

def get_node_from_row(row, num_nodes, max_label_length, max_energy=None):
    
    node_id = "node_"+str(row.name)
    node_name = str(row.name)
    x_coordinate = row.x
    y_coordinate = row.y
    if max_energy:
        size = row.energy * 50 / (max_energy * np.log(num_nodes))
    else:
        size = row.energy * 400 / np.log(num_nodes)
    text_x_coordinate = row.x * 1.5 + (np.sign(row.x) * 5)
    text_y_coordinate = row.y * 1.5
    font_size = max([13 - int(max_label_length/2), 7])
    label = str(row.name)
    
    return node_template.format(node_id, node_name, x_coordinate, y_coordinate, size, text_x_coordinate, text_y_coordinate, font_size, label)


edge_template = """<g id="{0}" class="edge"><title>{1}</title>
<line x1="{2:.2f}" y1="{3:.2f}" x2="{4:.2f}" y2="{5:.2f}" stroke="#D32F27" stroke-width="{6}"/>
</g>"""

def get_edge_from_row(row, num_nodes, max_engagement=None):
    
    edge_id = "edge_" + str(row.name[0]) + "_" + str(row.name[1])
    edge_name = str(row.name[0]) + "_" + str(row.name[1])
    x1 = row.x_1
    y1 = row.y_1
    x2 = row.x_2
    y2 = row.y_2
    if max_engagement:
        width = row.engagement * 100 / (max_engagement * num_nodes)
    else:
        width = row.engagement * 700 / num_nodes
    
    return edge_template.format(edge_id, edge_name, x1, y1, x2, y2, width)


preamble = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="420pt" height="450pt"
 viewBox="0.00 0.00 420.00 450.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1.0 1.0) rotate(0) translate(210 210)">
<title>G</title>"""


postamble_template = """<g id="time_annotation" class="node"><title>time_annotation</title>
<text text-anchor="middle" x="0" y="230" font-family="DejaVu Sans,sans-serif" font-size="13.00">{0}</text>
</g>
</g>
</svg>"""

def get_postamble(name):

    return postamble_template.format(name)


def visualise_energy_engagement(name, node_data, edge_data, num_nodes, max_label_length, max_energy=None, max_engagement=None):
    
    nodes = "\n".join(node_data.apply(get_node_from_row, axis=1, num_nodes=num_nodes, max_label_length=max_label_length, max_energy=max_energy).tolist())
    
    edges = "\n".join(edge_data.apply(get_edge_from_row, axis=1, num_nodes=num_nodes, max_engagement=max_engagement).tolist())
    
    postamble = get_postamble(name)
    
    return "\n".join([preamble, edges, nodes, postamble])


def plot_network_visualisations(critical_instabilities, per_second_speech, output_folder, time_zone, relative_scaling=True):
    ## kwargs used to pass stylistic adjustments to visualise_energy_engagement
    
    critical_times = np.concatenate([
        critical_instabilities.index[0:1],
        critical_instabilities[critical_instabilities].index,
        critical_instabilities.index[-1:]
    ])
    
    prev_end = pd.to_datetime("2000-01-01").tz_localize(time_zone)
    
    slice_names = []
    node_data = {}
    edge_data = {}
    max_label_length = max([len(label) for label in per_second_speech.columns])
    
    for start, end in zip(critical_times[:-1], critical_times[1:]):
        
        ## Only visualise if we have at least one minute within the slice
        if end - prev_end > np.timedelta64(60, "s"):
            
            slice_name = start.strftime("%H:%M:%S") + " - " + end.strftime("%H:%M:%S")
            slice_names.append(slice_name)
            
            node_data[slice_name] = get_node_data(per_second_speech[start:end])
            
            edge_data[slice_name] = get_edge_data(per_second_speech[start:end])
            
            prev_end = end
    
    max_energy = None
    max_engagement = None
    
    if relative_scaling:
    
        max_energy = max([slice.energy.max() for slice in node_data.values()])
        max_engagement = max([slice.engagement.max() for slice in edge_data.values()])
    
    for name in slice_names:
            
            name_reformatted = re.sub(":", "_", re.sub(" - ", "_to_", name))
            
            svg_string = visualise_energy_engagement(name, node_data[name], edge_data[name], num_nodes=len(per_second_speech.columns), max_label_length=max_label_length, max_energy=max_energy, max_engagement=max_engagement)
            
            node_data[name].energy.to_csv(os.path.join(output_folder, "energy_" + name_reformatted + ".csv"))
            edge_data[name].engagement.to_csv(os.path.join(output_folder, "engagement_" + name_reformatted + ".csv"))
            
            display(SVG(svg_string))
            
            with open(os.path.join(output_folder, name_reformatted + ".svg"), "w") as file:
                
                file.write(svg_string)


## This is just a simple workaround for now
def plot_network_overall(critical_instabilities, per_second_speech, output_folder, time_zone, relative_scaling=True):
    
    ## Set critical instabilities to false across the whole data set, so the data is not split up into multiple visualisations
    critical_instabilities = critical_instabilities & False
    
    plot_network_visualisations(critical_instabilities, per_second_speech, output_folder, time_zone, relative_scaling)