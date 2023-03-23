import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve2d,resample,detrend,butter,lfilter
from skimage.filters import threshold_otsu,threshold_multiotsu
from skimage import exposure
from tracking import tracking,define_tracking_region
from freqency import plot_fft_plus_power
from utilities import *
from external.gaussian_BP.linefitting import LineFittingModel

#Global settings
SIZE_X,SIZE_Y = 224,224
gradient = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
M = None #rotation matrix

def register_tracking_points(event,x,y,flags,param):
    global pts_queue
    if event == cv.EVENT_RBUTTONDOWN:
        pts_queue.append((x, y))


def OnInit(video_path):
    global cap, ROI, ptr_scores, p0
    global frame_count,LEFT_BOUND,RIGHT_BOUND 
    global M
    frame_count = 0; p0=None

    cap = cv.VideoCapture(video_path)
    if (cap.isOpened()== False):
      print(video_path)
      print("Error opening video stream or file")
      quit()
    size, fps = (int(cap.get(3)),int(cap.get(4))),cap.get(5)
    print(f"video size: {size}, fps: {fps}")

    print("-------------please select the RoI-------------")
    ret, frame = cap.read()
    cv.imshow('ROI selection',frame)
    #M,frame = Utilities.rotate(frame,'ROI selection')
    ROI = cv.selectROI(windowName='ROI selection',img=frame)
    print(ROI)
            
    bounding_box = cv.selectROI(windowName='ROI selection',img=frame)
    if bounding_box != (0,0,0,0):
        LEFT_BOUND,RIGHT_BOUND = int(bounding_box[0]),int(bounding_box[0]+bounding_box[2]-ROI[2])
        p0 = define_tracking_region(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), ROI)
        if p0 is not None:
            p0 = p0.reshape((p0.shape[0],2))
            ptr_scores = np.zeros((p0.shape[0]))
    else:
        print("Tracking disabled")

    cv.destroyAllWindows()
    return frame[:,:,0],size, fps

def InitModels(model_path,feature_extractors):
    global VGG_filters, model
    VGG_filters = keras.models.load_model(feature_extractors)
    try:
        model = pickle.load(open(model_path,'rb'))
    except:
        model = None
        print("Model is not defined")


def readframe(skip=0,as_gray=False):
    global cap, frame_count, M
    ret, frame = cap.read()
    if M is not None:
        frame = cv.warpAffine(frame, M, (frame.shape[0],frame.shape[1]))

    if not ret:
         print("End of video")
    if as_gray:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    frame_count = frame_count + skip + 1
    return ret, frame

def map1d(distance):
    global STmap
    if STmap is None:
        STmap = [np.float32(distance)]
    else:
        STmap.append(np.float32(distance))

def diamap(top,bottom):
    global sample_locations
    upper_mean = np.mean(top)
    lower_mean = np.mean(bottom)
    if sample_locations is None:
        sample_locations = [(upper_mean,lower_mean)]
    else:
        sample_locations.append((upper_mean,lower_mean))   


def plot(map_type = 1,filename=None):
    global time_scale,STmap,num_of_pts,pixel_scale_y,pixel_scale_x
    def remap(transistion_point,array):
        k1 = transistion_point[1]/transistion_point[0]
        k2 = (1-transistion_point[1])/(1-transistion_point[0])
        cutoff = transistion_point[0]
        return np.where(array<cutoff, k1*array,k2*array-k2+1)

    STmap = np.vstack(STmap)
    STmap_detrend = detrend(STmap,axis=0)
    d = np.median(STmap,axis=1)
    if filename is not None:
        data_to_save = {"STmap":STmap,
                        "mean_distance":d,
                        "scales":np.array([pixel_scale_x,pixel_scale_y]),
                        "breathing":breathing_movement,
                        "sample_rate": Fs}
        np.savez(filename,**data_to_save)

    fig1,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("diameter(cm)")
    ax2.set_xlabel("Frequency(s)")
    ax2.set_ylabel("Normalized PSD")
    ax1.plot(time_scale,d)
    plot_fft_plus_power(ax2,time_scale, detrend(d),True)
    DiaMap = np.ones((SIZE_X,len(sample_locations)),dtype=np.uint8)

    if map_type == 1:
        for col in range(DiaMap.shape[1]):
            y1,y2 = sample_locations[col]
            DiaMap[int(y1):int(y2),col] = 0
        plt.imshow(DiaMap,cmap='gray')
        plt.title("Diameter map")
        plt.axis('off')
        plt.show()
    else:
        b, a = butter(11, 0.3, fs=Fs, btype='low', analog=False)
        STmap = lfilter(b, a, STmap_detrend, axis=0)
        min_img = np.min(STmap)
        max_img = np.max(STmap)
        normalized = (STmap-min_img)/(max_img - min_img)
        p2, p98 = np.percentile(normalized, (2, 98))
        normalized = exposure.rescale_intensity(normalized, in_range=(p2, p98))
        normalized = remap((0.5,0.25),normalized)
        upscaled = resample(normalized,SIZE_Y,axis=1)
        yticks = ["{:.1f}".format(i) for i in pixel_scale_y * np.linspace(0,SIZE_Y,10)]
        ST_MAP = Interactive_frequencyplot(unit_time)
        ST_MAP.plot_stmap(upscaled,yticks)
        ST_MAP = Interactive_frequencyplot(unit_time)
        ST_MAP.plot_wavelet(STmap_detrend,d,wavelet_window,yticks)

def threhold(img_gray):
    img_gray = cv.bilateralFilter(img_gray,7,75,75)
    thres = threshold_otsu(img_gray)*0.9
    _, mask = cv.threshold(img_gray , thres, 255, cv.THRESH_BINARY)
    return mask

def threhold_multilevel(img_gray,thred=0.9):
    img_gray = cv.bilateralFilter(img_gray,7,75,75)
    thresholds = threshold_multiotsu(img_gray,classes=3)
    regions = np.digitize(img_gray, bins=thresholds).astype(np.float32)
    return cv.blur(regions,(7,3)) > thred
    
def init_edge(n_regions:int,mask:np.ndarray):
    global previous_1, previous_2
    sumh = np.sum(roberts(mask),axis=1)
    a,b = np.argmax(sumh[len(sumh)//2:])+len(sumh)//2,np.argmax(sumh[:len(sumh)//3])
    previous_1 = np.ones(n_regions,dtype=np.int32) * a
    previous_2 = np.ones(n_regions,dtype=np.int32) * b

    return previous_1, previous_2

def fast_smoothing(x:np.ndarray,max_iter:int=4):
    for i in range(max_iter):
        mid = np.median(x)
        mean = np.mean(x)
        std = np.std(x)
        x = np.where(abs(x-mean)<3*std,x,mid)
    return gaussian_filter1d(x,1)

def GBP_smoothing(x: np.ndarray,GBP_model:LineFittingModel, max_iter:int=5):
    return GBP_model.smooth(measuremnets=x,n_iters=max_iter)

def gather_samples(mask,cutoff):
    edge = convolve2d(mask, gradient,mode="valid")
    return edge[:cutoff,:]>0, edge[cutoff:,:]<0


#TO DO:
#add kalman filter procedure
def scanline_sampler(n_regions:int,mask:np.ndarray,k:float):
    global previous_1, previous_2, sigma1,sigma2
    increments = mask.shape[1]//n_regions
    c = mask.shape[0]//2
    m = int(max(min(c,np.mean(previous_1)-10),c/2))
    up_boundary,low_boundary = gather_samples(mask,m)
    j = 0

    def _local(samples,cutoff, ref, offset=0):
        max_size = min(10,len(samples))
        acm = 0
        count = 0
        weight=0
        for s in np.random.choice(samples[:,0],size=max_size):
            x = s+offset
            delta = abs(ref[i]-x)+1
            weight = (k/delta)**2
            if weight < cutoff[i]:
                weight = 1
                if np.random.uniform(low=0.0, high=1.0) < 0.1:
                    acm += x
                else:
                    acm += ref[i]
            else:
                acm += weight*x
            count += weight
        if count>1:
            cutoff[i] = min(0.01,cutoff[i]*2)
            ref[i] = acm/count
        else:
            ref[i] = ref[i] + np.random.uniform(low=-5,high=5)
            cutoff[i] = max(cutoff[i]/2,1e-6)        

    for i in range(n_regions):
        _local(np.argwhere(low_boundary[:,j:j+increments]),sigma1,previous_1,m)
        _local(np.argwhere(up_boundary[:,j:j+increments]),sigma2,previous_2,0)
        j+=increments

    sigma1 = gaussian_filter1d(sigma1,1)
    sigma2 = gaussian_filter1d(sigma2,1)
    return previous_1,previous_2

def measure_distance(lower_edge,upper_edge,sensitivity=5):
    global pixel_scale_x
    ret = np.zeros_like(lower_edge)
    for i in range(len(lower_edge)):
        dis = (lower_edge[i] - upper_edge[i])//sensitivity * sensitivity
        ret[i] = dis*pixel_scale_x
    return ret

def normpdf(x, mean, var=1):
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-mean)**2/(2*var))
    return num/denom

def mean_field_approx(img_input,iterations=2):
    img = np.where(img_input>0.5, img_input, img_input-1)
    mean = np.zeros_like(img)
    damping = 0.5
    field_strength = 10
    kernel = np.ones((5,5))
    kernel[2,2] = 0
    for i in range(iterations):
        field = convolve2d(img,kernel,mode='same')
        logOdds = np.log(normpdf(img,1))-np.log(normpdf(img,-1))
        mean = (1-damping)*mean + damping*np.tanh(field*field_strength + img*logOdds)        
    return mean>0

def segmentation(img_bgr,method,args=0.9):
    global VGG_filters, model

    if method == 'RF':
        img_input = cv.resize(img_bgr,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        img_input=cv.medianBlur(img_input,7)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = preprocess_input(img_input)
        features = VGG_filters.predict(img_input)
        x = features.reshape(-1,features.shape[-1])
        prob = (model.predict_proba(x)[:,1]).reshape((SIZE_X,SIZE_Y))
        mask = mean_field_approx(prob[4:-4,:],3)
    elif method == 'Multiotsu':
        img_input = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        img_input = cv.resize(img_input,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        mask = threhold_multilevel(img_input,thred=args)* np.uint8(255)
        mask = cv.medianBlur(mask,11)
    elif method == 'Otsu':
        img_input = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        img_input = cv.resize(img_input,(SIZE_X,SIZE_Y),interpolation=cv.INTER_NEAREST)
        mask = threhold(img_input)
        mask = cv.medianBlur(mask,11)

    return mask.astype(np.uint8)


def DEBUG_draw_sample(frame,n_regions):
    rescale_x = frame.shape[0]/SIZE_Y
    y_increment = frame.shape[1]/n_regions
    y = y_increment//2
    for i in range(len(previous_1)):
        x1,x2 = previous_1[i]*rescale_x, previous_2[i]*rescale_x
        cv.circle(frame,(int(y),int(x1)),1,(0,0,255),4)
        cv.circle(frame,(int(y),int(x2)),1,(0,0,255),4)
        y+=y_increment

if __name__ == "__main__":
#Parameters (EVERYTHING THAT NEEDS TO BE FILLED IN FOR CODE TO RUN)  
    video_path = r"E:\Haustral_motility_figures\Code\main\data\All_Figures\Maxwell_E15\C3\E15-TRANS-C3.mp4"
    map_type = 1            #1d: diameter map + freqency; 2d:2d color map 
    skip_frame = 2          #number of frame skipped
    num_of_pts = 10         #number of points for edge tracking
    smoothing = "FAST"       #GBP=slow FAST=fast approximation of walls
    wavelet_window = np.linspace(8,256,200)  #200 steps

#initialization
    model_path = r"E:\Haustral_motility_figures\Code\main\RF_model.sav"
    method = "Otsu"
    old_frame_gray,_,fps = OnInit(video_path)
    unit_time = (1+skip_frame)/fps
    Fs = 1/unit_time
    print(f"sample rate {Fs}")
    anchor = float(ROI[0])
    sample_locations =None
    analyzer = BinaryMaskAnalyser()

    if smoothing == "GBP":
        n_measurement = num_of_pts-1
        GBP1 = LineFittingModel(n_measurement, np.array([10.0]),np.array([10]),np.array([1]))
        GBP2 = LineFittingModel(n_measurement, np.array([10.0]),np.array([10]),np.array([1]))
    else:
        n_measurement = num_of_pts
    sigma1=np.ones(n_measurement);sigma2=sigma1.copy()


#scale, cm per pixel
    cm_per_pix = Utilities.get_pixscale(old_frame_gray,"draw 1cm box")
    pixel_scale_y = ROI[2]/(cm_per_pix*SIZE_Y)    
    pixel_scale_x = ROI[3]/(cm_per_pix*SIZE_X)

#collect data
    STmap = None
    breathing_movement = []
    pts_queue = []
    dx = 0
    init_state=True

    if method == "RF":
        if os.path.exists(model_path):
            from tensorflow import keras
            from keras.applications.vgg16 import preprocess_input
            from sklearn.ensemble import RandomForestClassifier
            InitModels(model_path,r"E:\Haustral_motility_figures\Code\main\dependencies\VGG_CONV_2")
        else:
            def feature_extractor(img):
                x_train = preprocess_input(img)
                features = VGG_filters.predict(x_train)
                return features.reshape(-1,features.shape[-1])

            cv.destroyAllWindows()
            temp = Segmenter(cap,ROI,SIZE_X,SIZE_Y,feature_extractor)
            model = temp.get_model()
            pickle.dump(model, open(model_path, 'wb'))

    cv.namedWindow('frame')
    cv.setMouseCallback('frame',register_tracking_points)
    while(1):
        ret, frame = readframe(skip_frame,as_gray=False)
        if ret:
            frame_gray = frame[:,:,0]
            key = cv.waitKey(1)
            if key & 0xff == 27:
                break 
            if key & 0xff == 97:#A
                anchor-=10
            if key & 0xff == 100:#D
                anchor+=10

            imCrop = frame[int(ROI[1]):int(ROI[1]+ROI[3]), int(anchor):int(anchor+ROI[2])] 
            mask = segmentation(imCrop,method,1.2)
            
            if p0 is not None:
                p0, ptr_scores,dx = tracking(frame_gray,old_frame_gray,p0,ptr_scores,len(pts_queue)!=0, pts_queue)
                for i,(x,y) in enumerate(p0):
                    cv.circle(frame,(int(x),int(y)),1,(255,0,0),4)         
                anchor = min(RIGHT_BOUND,max(LEFT_BOUND,dx+anchor))
            breathing_movement.append(anchor-ROI[0])    

            cv.rectangle(frame, (int(anchor),int(ROI[1])), (int(anchor+ROI[2]),int(ROI[1]+ROI[3])), (255,0,0),4)

            if init_state:
                analyzer.register_mask(mask)  
                mask = analyzer.refine_mask(4)
                previous_1, previous_2 = init_edge(n_measurement,mask)
            else:
                lower_edge, upper_edge = scanline_sampler(n_measurement,mask,k=10)
                if smoothing == "GBP":
                    lower_edge = GBP_smoothing(lower_edge,GBP1); upper_edge = GBP_smoothing(upper_edge,GBP2)
                    previous_1 = (lower_edge[1:] + lower_edge[:-1])/2
                    previous_2 = (upper_edge[1:] + upper_edge[:-1])/2
                else:
                    lower_edge = fast_smoothing(lower_edge); upper_edge = fast_smoothing(upper_edge)
                    previous_1 = lower_edge; previous_2 = upper_edge
                distance = measure_distance(lower_edge, upper_edge,1)
                diamap(upper_edge,lower_edge)
                map1d(distance)

            DEBUG_draw_sample(imCrop,num_of_pts)
            old_frame_gray = frame_gray.copy()
            init_state = False

            cv.imshow('frame', frame)
            cv.imshow('mask', mask)
        else:
            break

    cv.destroyAllWindows()
    cap.release()
    breathing_movement = np.array(breathing_movement) * pixel_scale_y
    time_scale = np.arange(0,len(STmap))*unit_time

    plot(map_type,"temp")
    fig3,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
    ax1.plot(breathing_movement)
    plt.title("breathing movement tracking")
    plot_fft_plus_power(ax2, time_scale, breathing_movement,True)