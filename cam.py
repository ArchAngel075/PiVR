import cv2;
import numpy as np;
import imutils;
import keyboard;
import math;
useSock = False;
global __IDENTITY__


FETCHIMAGE_MODE = {
    "CAM" : 0,
    "FILE" : 1,
    "HTTP" : 2,
}

MASK_COLORSPACE = {
    "RED" : [[[0, 70, 50],[10, 255, 255]],[[170, 70, 50],[180, 255, 255]]],
    "BLUE" : [[[100,150,0],[140,255,255]]],
    "GREEN" : [[[25, 52, 72],[102, 255, 255]]],
    # mask = cv2.inRange(img, np.array([0, 70, 50]), np.array([10, 255, 255])) #RED
    # mask = mask | cv2.inRange(img, np.array([170, 70, 50]), np.array([180, 255, 255])) #RED
    #mask = cv2.inRange(img, np.array([100,150,0]), np.array([140,255,255])) # BLUE
    # mask = cv2.inRange(img, np.array([25, 52, 72]), np.array([102, 255, 255])) # GREEN
}

def withMask(img,color=MASK_COLORSPACE["RED"]):
    mask = None;
    for grp_i in range(len(color)):
        grp = color[grp_i];
        low, high = grp[0], grp[1];
        if mask is None:
            mask = cv2.inRange(img, np.array(low), np.array(high))
        else:
            mask = mask | cv2.inRange(img, np.array(low), np.array(high));
    # mask = cv2.bitwise_and(img, img, mask = mask);
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    showWait(mask,"mask_debug",1)
    return mask;

def showWait(src,name,d=0):
    cv2.imshow(name,src);
    return cv2.waitKey(d);

def fetchNewImage(mode=FETCHIMAGE_MODE["CAM"], src=None):
    if(mode == FETCHIMAGE_MODE["HTTP"]):
        bytess = b'';
        imgBase = None;
        while True:
            bytess += src.read(8192*2)
            a = bytess.find(b'\xff\xd8')
            b = bytess.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytess[a:b+2]
                bytess = bytess[b+2:]
                if len(jpg) == 0:
                    continue
                imgBase = cv2.flip(cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), -1),1)
                # imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2HSV);
            if imgBase is None:
                continue;
            return imgBase;
    elif(mode == FETCHIMAGE_MODE["FILE"]):
        return cv2.imread(src);
    elif (mode == FETCHIMAGE_MODE["CAM"]):
        while True:
            ret, frame = src.read();
            if ret:
                return cv2.flip(frame,1);
            else:
                continue;


def point2Int(pt):
    return (int(pt[0]),int(pt[1]));

def findContour(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE);
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None;
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return {"center":center,"position":(x,y),"radius":radius}


#convert the given char to 4 bytes
def toByte4(ch,isChar=False):
    b = bytearray(4)
    if isChar == False:
        if(str.isdigit(ch)):
            as_int = int(ch)
            if as_int > 256:
                b[0] = as_int%256;
                b[1] = math.floor(as_int/256)
        else:
            return None;
    else:
        b[0] = ord(ch)
    return b


def pose2Packet(packet):
    dec = 3;
    #{TIXYSRGB}
    
    bytesList = []
    bytesList += toByte4(b'{',True);
    bytesList += toByte4(b'P',True);
    bytesList += toByte4(__IDENTITY__);

    bytesList += toByte4(math.floor(packet[0])); #X
    bytesList += toByte4(math.floor(packet[1])); #Y
    bytesList += toByte4(math.floor(packet[2])); #S
    
    bytesList += toByte4(255); #R
    bytesList += toByte4(0); #G
    bytesList += toByte4(0); #B

    bytesList += toByte4(b'}',True);
    
    return bytesList;

#process :
def process(src):
    print("begin process");
    z_origin = None;
    x_origin = None;
    y_origin = None;
    while True:
        if keyboard.is_pressed('f'):
            exit();
        best_keypoint = None;
        frame = cv2.flip(fetchNewImage(mode=FETCHIMAGE_MODE["CAM"],src=src),1);
        frame_result = frame_result if frame_result is not None else frame.copy();
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        working_frame = frame_hsv;
        working_frame = withMask(working_frame, MASK_COLORSPACE["RED"]);
        contour = findContour(working_frame); #None if failed to find -> we can drop state and scan entire frame in a new iter:
        #
        if contour is not None:
            best_keypoint = contour["center"]
            z_current = round(contour["radius"],3);
            x_current = round(best_keypoint[0],3);
            y_current = round(best_keypoint[1],3);
            frame_result = frame#drawPoints(frame,keypoints);
            frame_result = cv2.circle(frame_result, point2Int(contour["position"]), int(contour["radius"]), (0,0,255),2)
            if z_origin is None :
                z_origin = z_current #set the initial Z position first captured. this is origin 0
                #all new Z values will assume this is 0, the z position is thus the difference from this value
            if x_origin is None :
                x_origin = x_current #set the initial Z position first captured. this is origin 0
                #all new Z values will assume this is 0, the z position is thus the difference from this value
            if y_origin is None :
                y_origin = y_current #set the initial Z position first captured. this is origin 0
                #all new Z values will assume this is 0, the z position is thus the difference from this value
        else:
            continue;

        if keyboard.is_pressed('w') and not keyboard.is_pressed('q'):
            z_origin += 0.2;
        if keyboard.is_pressed('x') and not keyboard.is_pressed('q'):
            z_origin -= 0.2;

        if keyboard.is_pressed('a'):
            x_origin += 0.2;
        if keyboard.is_pressed('d'):
            x_origin -= 0.2;

        if keyboard.is_pressed('w') and keyboard.is_pressed('q'):
            y_origin += 0.2;
        if keyboard.is_pressed('x') and keyboard.is_pressed('q'):
            y_origin -= 0.2;

        if(best_keypoint is not None):
            frame_result = cv2.circle(frame_result, point2Int(best_keypoint), 3, (255,0,0), 2);
            frame_result = cv2.circle(frame_result, point2Int(best_keypoint), abs(int(z_current - z_origin)*6), (0,255,0), 2);
        vector = ((x_current - x_origin)*2 , (y_current - y_origin)*2 , (z_current - z_origin)*3 )
        frame_result = cv2.putText(frame_result,str(vector),(1,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0));
        cv2.imshow("color",frame_result);
        packet = pose2Packet(vector)
        # print("send socket packet : " + str(packet));
        return packet;
        # if useSock and best_keypoint is not None:
            #sock.sendall(bytearray(packet.encode()));