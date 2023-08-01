#!/usr/bin/env python3

#    yolo_tracker - sort video archives based on objects and their behavior
#    
#    Copyright (C) 2023  Alexander Gorodinski testor@zazoid.com
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.


############## we actually need to put it here to not load all cuda stuff before just showing "--help", ok?
import argparse
parser = argparse.ArgumentParser(description='Analyze video file(s) and make conclusion for the presence of objects of interest and how they moved.')
parser.add_argument('-a','--annotate', action='store_true', help='generate annotated video(draw boxes with params for detected objects)')
parser.add_argument('-m','--model', type=str, default='yolov8x.pt', help='default model is largest, specify a smaller model in case of memory or performance issues')
parser.add_argument('--input-type','-i', type=str, choices=['file', 'dir', 'inotify'], default='dir', \
                        help='process file(default), whole dir or utilize inotify for immediately process new files, right after their uploading is finished \n\
                        hint: use touch -m <that_file> if some file appeared at the moment script wasn\'t listening to inotify events')
parser.add_argument('path', type=str, default='./', nargs='?', help='Path to file or dir with files to process. Default is to process current dir')
args = parser.parse_args()


#import pdb
#import datetime
from ultralytics import YOLO
import cv2
import sys,os
import torch
import traceback
import math
import pprint
import numpy as np
import inotify.adapters


CONFIG = '/home/user/scripts/default.yaml'




model_name = args.model
path = args.path
annotate = args.annotate
input_type = args.input_type


#list of objects that we intrested in
#names can be found here:     ultralytics/ultralytics/cfg/datasets/coco.yaml
#TODO: convenient configuration
obj_classes_num = [0,1,2,3,7,16,15,24,25,26,28,39,56]

model = YOLO(model_name)

pp = pprint.PrettyPrinter(indent=4)

torch.cuda.empty_cache()

def process_video_file(video_cap):
   
    global annotate
    global f #for file matching in pdb cond debugging
    frames_processed = 0
                                                                        #oaky, maybe check what is it before.... 
    fps = video_cap.get(5)

    print(f"fps detected:{fps}")

    frames_with_detections = 0
    total_objects = 0

    target_obj_num = 0

    sum_conf = 0
    target_sum_conf = 0

    avg_conf = 0
    target_conf = 0
    target_conf_per_total_frames = 0

    objects = {}

    annotated_frames = []

    while True:

        ret, frame = video_cap.read()
        if not ret:
            #print(f"ret is: {ret}")
            break


        results = model.track(frame,cfg = CONFIG, verbose = False)


        j = 0
        for result in results:

            #fbackup = frame
            #frame = result.plot()
            no_id = False
            no_cls = False
            wrong_obj = False
            try:                                                            # there may be funny object
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                xywhs = result.boxes.xywh.cpu().numpy().astype(int)                 
                confs = result.boxes.conf.cpu().numpy().astype(float)
            except Exception as e:                                          # or suddenly not
                no_id = True                                                # be careful
                #print(traceback.format_exc())                              # and let that print remains just a comment for you
            try:
                cls = result.boxes.cls.cpu().numpy().astype(int)            # because you have pdb to uncomment
            except Exception as e:                                          # in imports
                no_cls = True                                               # but I eventually got used to -m pdb
            if no_id or no_cls:
                break
            try:
                for cl in cls:
                    if cl not in obj_classes_num: wrong_obj = True          # here are no giraffes in Belarus, for example
            except Exception as e:                                          # at least in a wild. but "person" and "car" are occuring
                print(traceback.format_exc())                               # unfortunately
            for box, xywh, id_num, cl, conf in zip(boxes, xywhs, ids, cls, confs):                                                            #so, take your prescripted substances
                #cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 1)
                #cv2.putText(frame,f"Id: {id_num}  x: {xywh[0]} y: {xywh[1]} ",(box[0], box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)   #don't like boxes
                col = 255*conf                                                                                                                #just look at this circle
                size = 10*conf                                                                                                                #at this pink circle
                cv2.circle(frame,(xywh[0],xywh[1]), int(3+size), (255,0,int(col)), int(size/3))                                               #how it pulses! and even slightly changes "col" component of cоlor
                #print(f"cl is {cl} and its name is {result.names[cl]}\n")                                                                    #hope you like if too
                if id_num not in objects:                                                                           #okay, let's begin creating an array
                    objects[id_num] = {}                                                                            #the array of madness
                if cl not in objects[id_num] and not wrong_obj:                                                     #now yours
                    objects[id_num][cl] = {'hits': 1, 'conf': conf, 'path': 0.0, "x": xywh[0], "y": xywh[1]}        #at first we create dicts of path ids, then populate it with dicts of classes 
                elif not wrong_obj:                                                                                 #classes may change, also you'll notice ids without classes at all, for example
                    objects[id_num][cl]['hits'] +=1                                                                 #we created cl with initial values, that may be incremented next time. or not.
                    objects[id_num][cl]['conf'] += conf                                                             #division of confidence/hits will return us something like an average conf later
                    if objects[id_num][cl]['hits']%int(fps) == 0:                                                   #every tenth hit we measure the length of the path traveled
                        #print('we see it for 10 frames, it\'s old enough to measure it\'s path')                   #this is first stage of filtering out the noise. Path isn't too curve or too straight.
                        prev = [objects[id_num][cl]['x'],objects[id_num][cl]['y']]                                  #10 is good sampling period to prevent parked cars collect path by integrating usual inference noise
                        now = [xywh[0],xywh[1]]                                                                     #it already points to center
                        dist = math.dist(prev,now)                                                                  #very simple, but
                        if dist < 4: dist = 0                                                                       #be very carefull, because pixels are probably integers!(did you check?)
                        elif dist >= 4 and int(dist) < 10: np.exp(int(dist)-6)/float("0.215e1")                     #this is purely empirical formula: noise filter
                        objects[id_num][cl]['path'] += dist                                                         #it's very sensitive to params
                        objects[id_num][cl]['x'] = xywh[0]                                                          #spent the night with matplotlib    
                        objects[id_num][cl]['y'] = xywh[1]                                                          #drawing something like beutiful "potential well"
                    cv2.putText(frame,f"{objects[id_num][cl]['path']:.0f} {id_num} {cl}",(box[0], box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2) #put ugly digits here to look like a freak
            j+=1                                                                                                                                      #ok, all these filters actually work great in my case
            frames_with_detections+=1                                                                                                                 #but what if you change the resolution?
        if j>1: print("MORE THAN ONE RESULT!!!")                                                                                                      #in short - don't do this. You need approx 640 of width
        if annotate:                                                                                                                                  #because Yolo anyway process in 680x480 afaik
            annotated_frames.append(frame)                                                                                                            #also my algo optimized for 10fps
        frames_processed+=1




    for id_num in objects:
        for cl in objects[id_num]:
            if cl == 0:                                                                                                                                                     #separately calculate significantly bigger coef for person
                #print('person!')                                                                                                                                           #person is still the most dangerous objects type
                objects[id_num][cl]['cl_coef'] = objects[id_num][cl]['conf']/objects[id_num][cl]['hits'] * (math.sqrt(objects[id_num][cl]['path']*objects[id_num][cl]['hits'])/25)      #that you may measure and calculate
            elif cl in obj_classes_num:
                objects[id_num][cl]['cl_coef'] = objects[id_num][cl]['conf']/objects[id_num][cl]['hits'] * (math.sqrt(objects[id_num][cl]['path']*objects[id_num][cl]['hits'])/100)     #for every track id we calculate class coef.
            else:                                                                                                                                                                       #the formula is totally empirical
                objects[id_num][cl]['cl_coef'] = 0                                                                                                                                      #for my needs (cctv in private household)
    
    top_cl_names = ""                                                                                                                                                                   #also: sometimes objects rob each other for id's
    top_id_coef = 0.0                                                                                                                                                                   #they say: "id was expropriated"                                                                         
    total_coef = 0.0       #so, now we finally have a three, my happy friends
    objects_top_class = -1 #so let's cycle over it, and collect all forbidden fruits of knowledge and sin *happy tree firends theme playng*
    top_cl_of_top_id = -1  #negatives here is to define the case of absense of any class, or any id, because 0, for example - is not the case of absense of any class.
    top_id = -1            #class id 0 is coresponds to "person"... do you see class interests behind any struggle by the way?
    numb_of_ids = 0        #you better don't, it's pretty dangerous! by the way num_id = 0 (it's path) doesnt exists
    cl_and_ids_num = 0     #but touching edge case in this algorithm... okay - do this
    for id_num in objects:
        max_coef = 0
        numb_of_ids +=1                                             # many numbers are no more than a funny numbers
        for cl in objects[id_num]:                                  # for fun and (hope not for the) furute usage
            this_cl_coef = objects[id_num][cl]['cl_coef']
            total_coef += this_cl_coef                              # right, we summ up all the coefs for the given object
            cl_and_ids_num += 1                                     # of the given class on it's whole path
            if this_cl_coef >= max_coef:                            # those who kept their identity over the loghest path with higher confidence
                max_coef = this_cl_coef                             # collect enough cl_coef to become the top class over all id's
                objects_top_class = cl                              # notice: max_coef is for every path id,
                obj_top_cls_name = result.names[objects_top_class]  #
        if objects_top_class > -1:                                  # why we need total_coef? ok, experiment shown that taking all classes into account,
            if max_coef >= top_id_coef:                             # even those who didn't won gives more realible result of detecting any activity of classes from list
                top_id_coef = max_coef                              # that why we intrested it and even treat as the most important. do we actually care who won? in fact - no.
                top_id = id_num                                     # the intensity of the struggle - that's what interests us, our task is to collect all possible metrics
                top_cl_of_top_id = objects_top_class                # for example top class for every id gives us useful summ string top_cl_names, which we add to the result file
            top_cl_names = top_cl_names +"_"+ obj_top_cls_name      # this is much more useful that the name of the winner, as long as total coeficient. other metrics are not really imortant
            print(f"top class for path id {id_num} is {obj_top_cls_name}")  #but looking over these prints actually helps to debug
        else:                                                       #this algo is actually the the for a happy firends, hope your meds are still working
            print(f"id {id_num} has no classes!!")                  #what to me, I made this algorithm work by simply tracing all possible cases in the debugger and conducting experiments on real data.
            #                                                       #the result is excellent. but if you have an idea how to understand it - let me know
            #                                                       #it was rewritten back and forth several times, and poor docs on yolo is another problem, but...
    if top_cl_of_top_id > -1:
        top_cl_name_of_top_id = result.names[top_cl_of_top_id]
        print(f"top classes are {top_cl_names}, top id is {top_id} and it's top class is {top_cl_name_of_top_id}")
        print(f"num of classes in every id is {cl_and_ids_num}")
        pp.pprint(objects)
    else:
        print(f"none of {numb_of_ids} id's has classes!!")          #classless id - it happens and it isn't rare case. probably confidence is below threshold over whole path
        top_cl_names = 'zero_detections'                            #useful file name and helps to avoid undef err

    #objects.clear()

    return total_coef,top_cl_names,annotated_frames


def write_annotated(annotated_frames,video_cap,dest_path):                                              #without ext!
                #print(f"starting of rendering {dest_path}.mp4")
                frame_size = (int(video_cap.get(3)),int(video_cap.get(4)))
                fps = video_cap.get(5)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                #you'll hate that functuin, I promise
                #print(f"params is, fourcc: {fourcc} fps: {fps} frame_size: {frame_size}")              #especially fourcc
                video_render = cv2.VideoWriter(dest_path+"_annotated.mp4",fourcc,fps,frame_size)        #also look, we don't use video_cap for writing
                for frame in annotated_frames:                                                          #just for determining some parameters
                    video_render.write(frame)
                    #print("frame written")
                video_render.release()


if input_type == 'dir':
    for filename in sorted(os.listdir(path)):
        f = os.path.join(path, filename)
        if os.path.isfile(f) and (os.path.splitext(f)[1] in ['.mp4','.mkv','.avi']):
            print(f"opening: {str(f)}")
            #try:
            video_cap = cv2.VideoCapture(f)
            coef, top_cl_names, annotated_frames = process_video_file(video_cap)
            #except Exception as e:
            #    coef = 0.0                                                             #func(), i know
            #    class_name = 'failed_to_process'
            #    print(traceback.format_exc())
            print(str(f)+" has coef.: "+str(coef)+" and top classses names are: "+str(top_cl_names)+" num of annotated frames: "+str(len(annotated_frames)))
            path,filename = os.path.split(f)
            filename,ext = os.path.splitext(filename)
            filename = filename+'_'+str(round(coef,3))+"_"+str(top_cl_names)
            if coef < 0.35:
                os.makedirs(path+"/delete/",exist_ok=True)
                dest_path = path+"/delete/"+filename            #without ext!
            elif coef >= 0.35 and coef < 0.65:
                os.makedirs(path+"/not_sure/",exist_ok=True)
                dest_path = path+"/not_sure/"+filename 
            elif coef >= 0.65 and coef < 1.8  :
                os.makedirs(path+"/less_confident/",exist_ok=True)
                dest_path = path+"/less_confident/"+filename+ext
            elif coef >= 1.8 and coef < 9.5:
                os.makedirs(path+"/confident/",exist_ok=True)
                dest_path = path+"/confident/"+filename
            else:
                os.makedirs(path+"/very_confident/",exist_ok=True)
                dest_path = path+"/very_confident/"+filename

            if annotate:
                write_annotated(annotated_frames,video_cap,dest_path) #without ext!
                video_cap.release()
                os.rename(f,dest_path+ext)
                print("done")

            else:
                video_cap.release()
                print(f"just copying {f} to {dest_path+ext}")
                os.rename(f,dest_path+ext)

elif input_type == 'file':
    if os.path.isfile(path) and os.path.splitext(path)[1]=='.mkv':
        video_cap = cv2.VideoCapture(path)
        coef, top_cl_names, annotated_frames = process_video_file(video_cap)
        dest_path = os.path.splitext(path)[0]+'_'+str(round(coef,3))+"_"+str(top_cl_names)
        if annotate:
            write_annotated(annotated_frames,video_cap,dest_path)                                               #without ext!
            video_cap.release()
            print("coef. is: "+str(coef)+" and top classses names are: "+str(top_cl_names)+" num of annotated frames: "+str(len(annotated_frames)))
        else:
            pass

elif input_type == 'inotify':
    
    i = inotify.adapters.Inotify()

    i.add_watch(path,mask=inotify.constants.IN_CLOSE_WRITE)

    for event in i.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        f = path+filename
        if(('timelapse' in f) or not (os.path.splitext(f)[1] in ['.mp4','.mkv','.avi'])):
            print(f"{str(f)} isn't something we need")
            break
        print(f"opening: {str(f)}")                                                     #that would be a function, but next time, okay?
        #try:
        video_cap = cv2.VideoCapture(f)
        coef, top_cl_names, annotated_frames = process_video_file(video_cap)
        #except Exception as e:
        #    coef = 0.0
        #    class_name = 'failed_to_process'
        #    print(traceback.format_exc())
        print(str(f)+" has coef.: "+str(coef)+" and top classses names are: "+str(top_cl_names)+" num of annotated frames: "+str(len(annotated_frames)))
        path,filename = os.path.split(f)                                               ###############right, splitting back, because we'll made it a function, do you remember?
        filename,ext = os.path.splitext(filename)
        filename = filename+'_'+str(round(coef,3))+"_"+str(top_cl_names)
        if coef < 0.35:
            os.makedirs(path+"/delete/",exist_ok=True)                                  #hope inotify will ignore THAT lol
            dest_path = path+"/delete/"+filename            #without ext!
        elif coef >= 0.35 and coef < 0.65:
            os.makedirs(path+"/not_sure/",exist_ok=True)
            dest_path = path+"/not_sure/"+filename 
        elif coef >= 0.65 and coef < 1.8  :
            os.makedirs(path+"/less_confident/",exist_ok=True)
            dest_path = path+"/less_confident/"+filename+ext
        elif coef >= 1.8 and coef < 9.5:
            os.makedirs(path+"/confident/",exist_ok=True)
            dest_path = path+"/confident/"+filename
        else:
            os.makedirs(path+"/very_confident/",exist_ok=True)
            dest_path = path+"/very_confident/"+filename

        if annotate:
            write_annotated(annotated_frames,video_cap,dest_path) #without ext!
            video_cap.release()
            os.rename(f,dest_path+ext)                                                  #but also create copy, because annotated and recompressed videos are just for debug
            print("done")

        else:
            video_cap.release()
            print(f"just copying {f} to {dest_path+ext}")
            os.rename(f,dest_path+ext)


