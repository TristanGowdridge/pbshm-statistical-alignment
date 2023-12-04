from flask import Blueprint, current_app, g, render_template, request, session, redirect, url_for, jsonify
from pbshm.authentication.authentication import authenticate_request
from pbshm.db import structure_collection
from datetime import datetime
from pytz import utc
from bokeh import colors
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend
from bokeh.models import CustomJSTickFormatter, BasicTicker
from bokeh.embed import components
from random import randint
import numpy as np
import json

from .models import NormalConditionAlignment

from collections import defaultdict
#Create the transfer learning Blueprint
bp = Blueprint(
    "transferlearningOffline",
    __name__,
    template_folder="templates"
)

# Convert datetime to nanoseconds since epoch
def datetime_to_nanoseconds_since_epoch(timestamp):
    delta = timestamp.astimezone(utc) - datetime.fromtimestamp(0, utc)
    return ((((delta.days * 24 * 60 * 60) + delta.seconds) * 1000000) + delta.microseconds) * 1000

# Convert nanoseconds since epoch to datetime
def nanoseconds_since_epoch_to_datetime(nanoseconds):
    return datetime.fromtimestamp(int(nanoseconds * 0.000000001), utc)

#Convert View
@bp.route("/convert/<int:nanoseconds>/<unit>")
def convert_nanoseconds(nanoseconds, unit):
    if unit == "microseconds": return str(int(nanoseconds * 0.001))
    elif unit == "milliseconds": return str(int(nanoseconds * 0.000001))
    elif unit == "seconds": return str(int(nanoseconds * 0.000000001))
    elif unit == "datetime": return datetime.fromtimestamp(int(nanoseconds * 0.000000001)).strftime("%Y-%m-%d %H:%M:%S")
    elif unit == "datetimeutc": return datetime.fromtimestamp(int(nanoseconds * 0.000000001), utc).strftime("%Y-%m-%d %H:%M:%S")
    raise Exception("Unsupported unit")

import os
#Details JSON View
@bp.route("/populations/<population>")
@authenticate_request("transferlearning-browse")
def population_details(population):
    populations=[]

    with open(f"pbshm/transferlearning/OfflineData/z24_data.json", "r") as z24_read, open(f"pbshm/transferlearning/OfflineData/s101_data.json", "r") as s101_read:
        z24_jsons = json.load(z24_read)
        s101_jsons = json.load(s101_read)

    populations.append(
        {
            'start': min(z24_jsons[0]["timestamp"], s101_jsons[0]["timestamp"]),
            'end': max(z24_jsons[-1]["timestamp"], s101_jsons[-1]["timestamp"]), 
            'structures': ["z24", "s101"],
            'name': 'transfer-learning-bridges',
            'channels': [
                {
                    'name': f'omega{i}',
                    'type': 'Natural Frequency',
                    'unit': 'Hz'
                }
                for i in range(1, 5)
            ]
        }
    )
    return jsonify(populations[0]) if len(populations) > 0 else jsonify()


# List View
@bp.route("/populations")
@authenticate_request("transferlearning-list")
def population_list(browse_endpoint="transferlearning.population_browse"):
    #Load All Populations
    with open(f"pbshm/transferlearning/OfflineData/z24_data.json", "r") as z24_read, open(f"pbshm/transferlearning/OfflineData/s101_data.json", "r") as s101_read:
        z24_jsons = json.load(z24_read)
        s101_jsons = json.load(s101_read)
    
    populations=[(
        {
            'name': 'transfer-learning-bridges',
            'start': nanoseconds_since_epoch_to_datetime(
                min(z24_jsons[0]["timestamp"], s101_jsons[0]["timestamp"])
            ).strftime("%Y-%m-%d %H:%M:%S"),
            'end': nanoseconds_since_epoch_to_datetime(
                max(z24_jsons[-1]["timestamp"], s101_jsons[-1]["timestamp"])
            ).strftime("%Y-%m-%d %H:%M:%S"), 
            'structures': ["z24", "s101"],
            'browse': '/transferlearning/populations/transfer-learning-bridges/browse'
        }
    )]
    
    return render_template("list.html", populations=populations)


# Browse View
left_storage = {
    "structuresLeft":[],
    "channelsLeft":[],
    "scriptsLeft":None,
    "figureLeft":None,
    "start_timeLeft":"",
    "start_dateLeft":"",
    "end_timeLeft":"",
    "end_dateLeft":""
}
right_storage = {
    "structuresRight":[],
    "channelsRight":[],
    "scriptsRight":None,
    "figureRight":None,
    "start_timeRight":"",
    "start_dateRight":"",
    "end_timeRight":"",
    "end_dateRight":""
}
left_data = {}
right_data = {}
@bp.route("/populations/<population>/browse", methods=("GET", "POST"))
@authenticate_request("transferlearning-browse")
def population_browse(population):
    #Load All Populations
    populations=[]
    
    #Handle Request
    error, jsLeft, htmlLeft, structuresLeft, channelsLeft=None, None, None, [], []
    error, jsRight, htmlRight, structuresRight, channelsRight=None, None, None, [], []
    
    htmlBefore, jsBefore = None, None
    htmlAfter, jsAfter = None, None

    if request.method == "POST":
        form_id = request.form["form-id"]

        if form_id == "leftForm":
            #Validate Inputs
            startDateLeft = request.form["start-date-left"]
            startTimeLeft = request.form["start-time-left"]
            endDateLeft = request.form["end-date-left"]
            endTimeLeft = request.form["end-time-left"]
            structuresLeft = request.form.getlist("structures-left")
            channelsLeft = request.form.getlist("channels-left")
            startDateLeftParts = [int(part) for part in startDateLeft.split("-")] if startDateLeft else []
            startTimeLeftParts = [int(part) for part in startTimeLeft.split(":")] if startTimeLeft else []
            endDateLeftParts = [int(part) for part in endDateLeft.split("-")] if endDateLeft else []
            endTimeLeftParts = [int(part) for part in endTimeLeft.split(":")] if endTimeLeft else []
            if len(startDateLeftParts) != 3: error = "Left Start date not in yyyy-mm-dd format."
            elif len(startTimeLeftParts) != 2: error = "Left Start time not in hh:mm format."
            elif len(endDateLeftParts) != 3: error = "Left End date not in yyyy-mm-dd format."
            elif len(endTimeLeftParts) != 2: error = "Left End time not in hh:mm format."
            #Process request if no errors
            if error is None:
                #Create Match and Project aggregate steps
                startTimestampLeft = datetime_to_nanoseconds_since_epoch(datetime(startDateLeftParts[0], startDateLeftParts[1], startDateLeftParts[2], startTimeLeftParts[0], startTimeLeftParts[1]))
                endTimestampLeft = datetime_to_nanoseconds_since_epoch(datetime(endDateLeftParts[0], endDateLeftParts[1], endDateLeftParts[2], endTimeLeftParts[0], endTimeLeftParts[1]))
                
                document_xLeft, document_yLeft, document_colorLeft = defaultdict(list), defaultdict(list), {}
                for structureLeft in structuresLeft:
                    with open(f"pbshm/transferlearning/OfflineData/{structureLeft}_data.json", "r") as json_read:
                        jsons_loaded = json.load(json_read)

                    for document in jsons_loaded:
                        if not (startTimestampLeft < document["timestamp"] < endTimestampLeft):
                            continue
                        
                        for channel in document["channels"]:
                            if channel["name"] not in channelsLeft:
                                continue

                            if isinstance(channel["value"], dict):
                                for key in channel["value"]:
                                    name = f"{document['name']} - {channel['name']} ({key})"
                                    if name not in document_xLeft:
                                        document_colorLeft[name] = colors.named.__all__[randint(0, len(colors.named.__all__) - 1)]
                                    document_xLeft[name].append(document["timestamp"])
                                    document_yLeft[name].append(channel["value"][key])
                                    
                            elif isinstance(channel["value"], (int, float)):
                                name = f"{document['name']} - {channel['name']}"
                                if name not in document_xLeft:
                                    document_colorLeft[name] = colors.named.__all__[randint(0, len(colors.named.__all__) - 1)]
                                document_xLeft[name].append(document["timestamp"])
                                document_yLeft[name].append(channel["value"])                              
                
                #Create figure
                figLeft = figure(
                    tools="pan,box_zoom,reset,save",
                    output_backend="webgl",
                    height=300,
                    sizing_mode="scale_width",
                    title="Population: {population} Structures: {structuresLeft} Channels: {channelsLeft}".format(
                        population=population,
                        structuresLeft=', '.join(structuresLeft) if structuresLeft else "All",
                        channelsLeft=', '.join(channelsLeft) if channelsLeft else "All"
                    ),
                    x_axis_label="Time",
                )
                figLeft.toolbar.logo=None
                figLeft.toolbar.autohide=True
                for line in document_xLeft:
                    figLeft.line(document_xLeft[line], document_yLeft[line], line_color=document_colorLeft[line], legend_label=line)

                figLeft.xaxis.formatter = CustomJSTickFormatter(code="""
                    //DateTime Utilities
                    function pad(number, padding) { return number.toString().padStart(padding, '0'); }
                    function convertNanoseconds(nanoseconds) {
                        const milliseconds = Math.floor(nanoseconds / 1e6);
                        const remainderNanoseconds = nanoseconds - (milliseconds * 1e6);
                        return { milliseconds, remainderNanoseconds };
                    }
                    //Process Current Tick
                    const {milliseconds, remainderNanoseconds} = convertNanoseconds(tick);         
                    var date = new Date(milliseconds);
                    var formattedSubSeconds = ((remainderNanoseconds > 0) ? "." + pad(date.getMilliseconds(), 3) + remainderNanoseconds.toString().padStart(6, '0') : (date.getMilliseconds() > 0) ? "." + pad(date.getMilliseconds(), 3) : '');
                    return pad(date.getDate(), 2) + '/' + pad(date.getMonth() + 1, 2) + '/' + date.getFullYear() + " " + pad(date.getHours(), 2) + ':' + pad(date.getMinutes(), 2) + ':' + pad(date.getSeconds(), 2) + formattedSubSeconds;
                """)
                figLeft.xaxis.major_label_orientation = 3.14159264 / 2
                figLeft.xaxis.ticker = BasicTicker(desired_num_ticks=15)
                jsLeft, htmlLeft=components(figLeft)

                global left_storage
                left_storage = {
                    "structuresLeft":structuresLeft,
                    "channelsLeft":channelsLeft,
                    "scriptsLeft":jsLeft,
                    "figureLeft":htmlLeft,
                    "start_timeLeft":startTimeLeft,
                    "start_dateLeft":startDateLeft,
                    "end_timeLeft":endTimeLeft,
                    "end_dateLeft":endDateLeft
                }
        
        elif form_id == "rightForm":
            startDateRight = request.form["start-date-right"]
            startTimeRight = request.form["start-time-right"]
            endDateRight = request.form["end-date-right"]
            endTimeRight = request.form["end-time-right"]
            structuresRight = request.form.getlist("structures-right")
            channelsRight = request.form.getlist("channels-right")
            startDateRightParts = [int(part) for part in startDateRight.split("-")] if startDateRight else []
            startTimeRightParts = [int(part) for part in startTimeRight.split(":")] if startTimeRight else []
            endDateRightParts = [int(part) for part in endDateRight.split("-")] if endDateRight else []
            endTimeRightParts = [int(part) for part in endTimeRight.split(":")] if endTimeRight else []
            if len(startDateRightParts) != 3: error = "Right Start date not in yyyy-mm-dd format."
            elif len(startTimeRightParts) != 2: error = "Right Start time not in hh:mm format."
            elif len(endDateRightParts) != 3: error = "Right End date not in yyyy-mm-dd format."
            elif len(endTimeRightParts) != 2: error = "Right End time not in hh:mm format."
            #Process request if no errors
            if error is None:
                #Create Match and Project aggregate steps
                startTimestampRight = datetime_to_nanoseconds_since_epoch(datetime(startDateRightParts[0], startDateRightParts[1], startDateRightParts[2], startTimeRightParts[0], startTimeRightParts[1]))
                endTimestampRight = datetime_to_nanoseconds_since_epoch(datetime(endDateRightParts[0], endDateRightParts[1], endDateRightParts[2], endTimeRightParts[0], endTimeRightParts[1]))
                
                document_xRight, document_yRight, document_colorRight = defaultdict(list), defaultdict(list), {}
                for structureRight in structuresRight:
                    with open(f"pbshm/transferlearning/OfflineData/{structureRight}_data.json", "r") as json_read:
                        jsons_loaded = json.load(json_read)

                    for document in jsons_loaded:
                        if not (startTimestampRight < document["timestamp"] < endTimestampRight):
                            continue
                        
                        for channel in document["channels"]:
                            if channel["name"] not in channelsRight:
                                continue

                            if isinstance(channel["value"], dict):
                                for key in channel["value"]:
                                    name = f"{document['name']} - {channel['name']} ({key})"
                                    if name not in document_xRight:
                                        document_colorRight[name] = colors.named.__all__[randint(0, len(colors.named.__all__) - 1)]
                                    document_xRight[name].append(document["timestamp"])
                                    document_yRight[name].append(channel["value"][key])
                                    
                            elif isinstance(channel["value"], (int, float)):
                                name = f"{document['name']} - {channel['name']}"
                                if name not in document_xRight:
                                    document_colorRight[name] = colors.named.__all__[randint(0, len(colors.named.__all__) - 1)]
                                document_xRight[name].append(document["timestamp"])
                                document_yRight[name].append(channel["value"])                              
                
                #Create figure
                figRight = figure(
                    tools="pan,box_zoom,reset,save",
                    output_backend="webgl",
                    height=300,
                    sizing_mode="scale_width",
                    title="Population: {population} Structures: {structuresRight} Channels: {channelsRight}".format(
                        population=population,
                        structuresRight=', '.join(structuresRight) if structuresRight else "All",
                        channelsRight=', '.join(channelsRight) if channelsRight else "All"
                    ),
                    x_axis_label="Time",
                )
                figRight.toolbar.logo=None
                figRight.toolbar.autohide=True
                global right_data
                for line in document_xRight:
                    figRight.line(document_xRight[line], document_yRight[line], line_color=document_colorRight[line], legend_label=line)
                
                figRight.xaxis.formatter = CustomJSTickFormatter(code="""
                    //DateTime Utilities
                    function pad(number, padding) { return number.toString().padStart(padding, '0'); }
                    function convertNanoseconds(nanoseconds) {
                        const milliseconds = Math.floor(nanoseconds / 1e6);
                        const remainderNanoseconds = nanoseconds - (milliseconds * 1e6);
                        return { milliseconds, remainderNanoseconds };
                    }
                    //Process Current Tick
                    const {milliseconds, remainderNanoseconds} = convertNanoseconds(tick);         
                    var date = new Date(milliseconds);
                    var formattedSubSeconds = ((remainderNanoseconds > 0) ? "." + pad(date.getMilliseconds(), 3) + remainderNanoseconds.toString().padStart(6, '0') : (date.getMilliseconds() > 0) ? "." + pad(date.getMilliseconds(), 3) : '');
                    return pad(date.getDate(), 2) + '/' + pad(date.getMonth() + 1, 2) + '/' + date.getFullYear() + " " + pad(date.getHours(), 2) + ':' + pad(date.getMinutes(), 2) + ':' + pad(date.getSeconds(), 2) + formattedSubSeconds;
                """)
                figRight.xaxis.major_label_orientation = 3.14159264 / 2
                figRight.xaxis.ticker = BasicTicker(desired_num_ticks=15)
                jsRight, htmlRight=components(figRight)
                
                global right_storage
                right_storage = {
                    "structuresRight":structuresRight,
                    "channelsRight":channelsRight,
                    "scriptsRight":jsRight,
                    "figureRight":htmlRight,
                    "start_timeRight":startTimeRight,
                    "start_dateRight":startDateRight,
                    "end_timeRight":endTimeRight,
                    "end_dateRight":endDateRight
                }

        elif form_id == "transfer":
            # Process Source Data
            startDateLeft = left_storage["start_dateLeft"]
            startTimeLeft = left_storage["start_timeLeft"]
            endDateLeft = left_storage["end_dateLeft"]
            endTimeLeft = left_storage["end_timeLeft"]
            startDateLeftParts = [int(part) for part in startDateLeft.split("-")] if startDateLeft else []
            startTimeLeftParts = [int(part) for part in startTimeLeft.split(":")] if startTimeLeft else []
            endDateLeftParts = [int(part) for part in endDateLeft.split("-")] if endDateLeft else []
            endTimeLeftParts = [int(part) for part in endTimeLeft.split(":")] if endTimeLeft else []
            startTimestampLeft = datetime_to_nanoseconds_since_epoch(datetime(startDateLeftParts[0], startDateLeftParts[1], startDateLeftParts[2], startTimeLeftParts[0], startTimeLeftParts[1]))
            endTimestampLeft = datetime_to_nanoseconds_since_epoch(datetime(endDateLeftParts[0], endDateLeftParts[1], endDateLeftParts[2], endTimeLeftParts[0], endTimeLeftParts[1]))
            with open(f"pbshm/transferlearning/OfflineData/{left_storage['structuresLeft'][0]}_data.json", "r") as source_file:
                source_data = json.load(source_file)
                source_natfreqs = []
                source_timestamps = []
                source_labels = []
                for datum in source_data:
                    source_timestamps.append(datum["timestamp"])
                    
                    temp = [np.nan]*4  # Hard codes as 4 nat freqs in z24
                    for channel in datum["channels"]:
                        if channel["name"] == "label":
                            temporary_label = channel["value"]
                            if startTimestampLeft <= datum["timestamp"] <= endTimestampLeft:
                                temporary_label = 0
                            source_labels.append(temporary_label)
                        else:
                            natfreq_index = int(channel["name"][-1])-1
                            temp[natfreq_index] = channel["value"]
                        
                    source_natfreqs.append(temp)

                Xs = np.array(source_natfreqs)
                source_indices_nonans = np.where(~np.isnan(Xs).any(axis=1))[0]
                Xs = Xs[source_indices_nonans]
                source_timestamps = np.array(source_timestamps)[source_indices_nonans]

                ys = np.array(source_labels)
                ys = ys[source_indices_nonans]
            
            # Process Target Data
            startDateRight = right_storage["start_dateRight"]
            startTimeRight = right_storage["start_timeRight"]
            endDateRight = right_storage["end_dateRight"]
            endTimeRight = right_storage["end_timeRight"]
            startDateRightParts = [int(part) for part in startDateRight.split("-")] if startDateRight else []
            startTimeRightParts = [int(part) for part in startTimeRight.split(":")] if startTimeRight else []
            endDateRightParts = [int(part) for part in endDateRight.split("-")] if endDateRight else []
            endTimeRightParts = [int(part) for part in endTimeRight.split(":")] if endTimeRight else []
            startTimestampRight = datetime_to_nanoseconds_since_epoch(datetime(startDateRightParts[0], startDateRightParts[1], startDateRightParts[2], startTimeRightParts[0], startTimeRightParts[1]))
            endTimestampRight = datetime_to_nanoseconds_since_epoch(datetime(endDateRightParts[0], endDateRightParts[1], endDateRightParts[2], endTimeRightParts[0], endTimeRightParts[1]))
            with open(f"pbshm/transferlearning/OfflineData/{right_storage['structuresRight'][0]}_data.json", "r") as target_file:
                target_data = json.load(target_file)
                target_natfreqs = []
                target_timestamps = []
                target_labels = []
                for datum in target_data:
                    target_timestamps.append(datum["timestamp"])
                    temp = [np.nan]*5  # Hard codes as 5 nat freqs in s101.
                    for channel in datum["channels"]:
                        if channel["name"] == "label":
                            temporary_label = channel["value"]
                            if startTimestampRight <= datum["timestamp"] <= endTimestampRight:
                                temporary_label = 0
                            target_labels.append(temporary_label)
                        else:
                            natfreq_index = int(channel["name"][-1])-1
                            temp[natfreq_index] = channel["value"]
                        
                    target_natfreqs.append(temp)

                Xt = np.array(target_natfreqs)
                target_indices_nonans = np.where(~np.isnan(Xt[:, :-1]).any(axis=1))[0]
                Xt = Xt[target_indices_nonans]
                target_timestamps = np.array(target_timestamps)[target_indices_nonans]

                yt = np.array(target_labels)
                yt = yt[target_indices_nonans]

            minimum_modes = min(Xs.shape[1], Xt.shape[1])
            Xs = Xs[:, :minimum_modes]
            Xt = Xt[:, :minimum_modes]
            
            # Do the domain adapatation
            nca = NormalConditionAlignment()
            Xs_nca, Xt_nca = nca.fit_transform(Xs, Xt, ys, yt)


            # Plot Before domain adaptation
            col=["red", "blue", "green", "magenta"]
            figBefore = figure(
                title="Before Domain Adaptation",
                tools="pan,box_zoom,reset,save",
                output_backend="webgl",
                height=300,
                sizing_mode="scale_width"
            )
            
            legend_itemsBefore = []
            for i in range(1):  # Jack's code plotted multiple figures, I only want the first. This was the reasoning for the plots looking wierd in the call.
                # Plot the source data.
                for y in np.unique(ys):
                    indices = np.where(ys == y)
                    source = ColumnDataSource(data={
                        'x': Xs[indices, i].flatten(),
                        'y': Xs[indices, i + 1].flatten()
                    })
                    r = figBefore.scatter('x', 'y', source=source, color=col[int(y)], marker='circle')
                    if i == 0:
                        legend_itemsBefore.append((f"Source {int(y)}", [r]))

                # Plot the target data.
                for y in np.unique(yt):
                    indices = np.where(yt == y)
                    target = ColumnDataSource(data={
                        'x': Xt[indices, i].flatten(),
                        'y': Xt[indices, i + 1].flatten()
                    })
                    r = figBefore.scatter('x', 'y', source=target, color=col[int(y)], marker='x')
                    if i == 0:
                        legend_itemsBefore.append((f"Target {int(y)}", [r]))
                
            figBefore.toolbar.logo=None
            figBefore.toolbar.autohide=True
            figBefore.xaxis.axis_label = f"X_{i+1}"
            figBefore.yaxis.axis_label = f"X_{i+2}"

            # Create and add legend
            legend = Legend(items=legend_itemsBefore)
            figBefore.add_layout(legend)
            figBefore.legend.location = 'bottom_right'
            jsBefore, htmlBefore = components(figBefore)
            
            
            # Plot After domain adaptation
            col=["red", "blue", "green", "magenta"]
            figAfter = figure(
                title="After Domain Adaptation",
                tools="pan,box_zoom,reset,save",
                output_backend="webgl",
                height=300,
                sizing_mode="scale_width"
            )
            
            legend_itemsAfter = []
            for i in range(1):  # Jack's code plotted multiple figures, I only want the first. This was the reasoning for the plots looking wierd in the call.
                # Plot the source data.
                for y in np.unique(ys):
                    indices = np.where(ys == y)
                    source = ColumnDataSource(data={
                        'x': Xs_nca[indices, i].flatten(),
                        'y': Xs_nca[indices, i + 1].flatten()
                    })
                    r = figAfter.scatter('x', 'y', source=source, color=col[int(y)], marker='circle')
                    if i == 0:
                        legend_itemsAfter.append((f"Source {int(y)}", [r]))

                # Plot the target data.
                for y in np.unique(yt):
                    indices = np.where(yt == y)
                    target = ColumnDataSource(data={
                        'x': Xt_nca[indices, i].flatten(),
                        'y': Xt_nca[indices, i + 1].flatten()
                    })
                    r = figAfter.scatter('x', 'y', source=target, color=col[int(y)], marker='x')
                    if i == 0:
                        legend_itemsAfter.append((f"Target {int(y)}", [r]))
                
            figAfter.toolbar.logo=None
            figAfter.toolbar.autohide=True
            figAfter.xaxis.axis_label = f"$X_{i+1}$"
            figAfter.yaxis.axis_label = f"$X_{i+2}$"

            # Create and add legend
            legend = Legend(items=legend_itemsAfter)
            figAfter.add_layout(legend)
            figAfter.legend.location = 'bottom_right'
            jsAfter, htmlAfter = components(figAfter)
    
    #Render Template
    return render_template(
        "browse_new.html", error=error, population=population, populations=[population],
        **left_storage,
        **right_storage,
        figureBefore=htmlBefore, scriptsBefore=jsBefore,
        figureAfter=htmlAfter, scriptsAfter=jsAfter,
        )