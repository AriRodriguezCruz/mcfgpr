# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#django
# - - -
#python
import cv2
from pyModelChecking import *
from pyModelChecking.LTL import *
from pyModelChecking.CTL import *
import os
import numpy as np
from . import ransac
from . import ClassyVirtualReferencePoint as ClassyVirtualReferencePoint
from decimal import Decimal
import threading
#gazepattern
from eyedetector.models import XYPupilFrame, ExperimentPoint
from .application import ShowImage

BASE_DIR = os.path.dirname(__file__)

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class fulfills the model interface needed by the ransac() function.

    """
    # lists of indices of input and output columns
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
##        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
##        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
##        x,resids,rank,s = scipy.linalg.lstsq(A,B)
##        return x
        HT = np.linalg.lstsq(data[:,self.input_columns], data[:,self.output_columns])[0] # returns a tuple, where index 0 is the solution matrix.
        return HT

    def get_error( self, data, model):
        B_fit = data[:,self.input_columns].dot(model)
        err_per_point = np.sum((data[:,self.output_columns]-B_fit)**2,axis=1) # sum squared error per row
        err_per_point = np.sqrt(err_per_point) # I'll see if this helps. If not remove for speed.
        return err_per_point

class CheckCamera(object):

    HAARFACECASCADE = cv2.CascadeClassifier(BASE_DIR + "/haarcascades/haarcascade_frontalface_alt.xml")
    HAAREYECASCADE = cv2.CascadeClassifier(BASE_DIR + "/haarcascades/haarcascade_eye.xml")
    WINDOW_NAME = "Verificar camara."
    VERBOSE = True
    BLOWUP_FACTOR = 1 # Resizes image before doing the algorithm. Changing to 2 makes things really slow. So nevermind on this.
    #RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
    DILATIONWIDTH = 1+2*BLOWUP_FACTOR #must be an odd number
    DILATIONHEIGHT = 1+2*BLOWUP_FACTOR #must be an odd number
    DILATIONKERNEL = np.ones((DILATIONHEIGHT,DILATIONWIDTH),'uint8')
    WRITEEYEDEBUGIMAGES = False #enable to export image files showing pupil center probability

    pupilspacingrunningavg = None
    warm = 0
    offset_running_avg = None
    show_main_image = True
    virtualpoint = None

    def __init__(self, *args, **kwargs):
        cv2.namedWindow(self.WINDOW_NAME) # open a window to show debugging images
        vc = cv2.VideoCapture(0) # Initialize the default camera
        self.make_detector()
        try:
            if vc.isOpened(): # try to get the first frame
                (readSuccessful, frame) = vc.read()
            else:
                raise(Exception("failed to open camera."))
                readSuccessful = False
        
            while readSuccessful:
                pupilOffsetXYList = self.get_offset(frame, allowDebugDisplay=True)
                key = cv2.waitKey(10)
                if key == 27: # exit on ESC
                    #cv2.imwrite( "lastOutput.png", frame) #save the last-displayed image to file, for our report
                    cv2.destroyAllWindows()
                    #mainForTraining()
                    break
                # Get Image from camera
                readSuccessful, frame = vc.read()
        finally:
            vc.release() #close the camera
            #cv2 class="destroyWin"></cv2>dow(self.WINDOW_NAME) #close the window

    def get_offset(self, frame, allowDebugDisplay=True, trackAverageOffset=True, directInferenceLeftRight=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # find faces and eyes
        minFaceSize = (80,80)
        minEyeSize = (25,25)
        faces = self.detect(gray,self.HAARFACECASCADE,minFaceSize)
        eyes = self.detect(gray,self.HAAREYECASCADE,minEyeSize)
        drawKeypoints = allowDebugDisplay #can set this false if you don't want the keypoint ID numbers
        if allowDebugDisplay:
            output = frame
            self.draw_rects(output,faces,(0,255,0)) #BGR format
        else:
            output = None
            ##        draw_rects(output,eyes,(255,0,0))
        leftEye_rightEye = self.get_left_and_right_eyes( faces, eyes)
        if leftEye_rightEye: #if we found valid eyes in a face
            ##            draw_rects(output,leftEye_rightEye,(0,0,255)) #BGR format
            xDistBetweenEyes = (leftEye_rightEye[0][0]+leftEye_rightEye[0][1]+leftEye_rightEye[1][0]+leftEye_rightEye[1][1])/4 #for debugging reference point
            pupilXYList = []
            pupilCenterEstimates = []
            for eyeIndex, eye in enumerate(leftEye_rightEye):
            ##        eye = leftEye_rightEye[1]
                corner = eye.copy()

                #eyes are arrays of the form [minX, minY, maxX, maxY]
                eyeWidth = eye[2]-eye[0]
                eyeHeight = eye[3]-eye[1]
                eye[0] += eyeWidth*.20
                eye[2] -= eyeWidth*.15
                eye[1] += eyeHeight*.3
                eye[3] -= eyeHeight*.2
                eye = np.round(eye)
                eyeImg = gray[eye[1]:eye[3], eye[0]:eye[2]]
                if directInferenceLeftRight:
                    (cy,cx, centerProb) = self.get_pupil_center(eyeImg, True)
                    pupilCenterEstimates.append(centerProb.copy())
                else:
                    (cy,cx) = self.get_pupil_center(eyeImg, True)
                pupilXYList.append( (cx+eye[0],cy+eye[1])  )
                if allowDebugDisplay:
                    cv2.rectangle(output, (eye[0], eye[1]), (eye[2], eye[3]), (0,255,0), 1)
                    y = int(pupilXYList[eyeIndex][0])
                    x = int(pupilXYList[eyeIndex][1])
                    cv2.circle(output, (y,x),  3, (255,0,0),thickness=1) #BGR format


            # tear-duct of the camera-right eye
            ##        corner[0] += eyeWidth*0
            ##        corner[2] -= eyeWidth*.6
            ##        corner[1] += eyeHeight*.4
            ##        corner[3] -= eyeHeight*.35
            ##        corner = np.round(corner)
            ##        cv2.rectangle(output, (corner[0], corner[1]), (corner[2], corner[3]), (0,0,0), 1)
            ##        cornerImg = gray[corner[1]:corner[3], corner[0]:corner[2]]
            ##        (cornerCy,cornerCx) = getEyeCorner(cornerImg)
            ##        cv2.circle(output, (cornerCx+corner[0],cornerCy+corner[1]), 2, (255,255,0),thickness=1) #BGR format

            # direct inference combination of the two eye probability images.
            #global PupilSpacingRunningAvg
            if directInferenceLeftRight:
                # these vectors are in XY format
                pupilSpacing = np.array(pupilXYList[1])-np.array(pupilXYList[0]) # vector from pupil 0 to pupil 1
                if self.pupilspacingrunningavg is None:
                    self.pupilspacingrunningavg = pupilSpacing
                else:
                    weightOnNew = .03
                    self.pupilspacingrunningavg = (1-weightOnNew)*self.pupilspacingrunningavg + weightOnNew*pupilSpacing  # vector from pupil 0 to pupil 1
                if allowDebugDisplay:
                    cv2.line(output, (int(pupilXYList[0][0]),int(pupilXYList[0][1])), (int(pupilXYList[0][0]+self.pupilspacingrunningavg[0]), int(pupilXYList[0][1]+self.pupilspacingrunningavg[1])), (0,100,100))
                imageZeroToOneVector = leftEye_rightEye[1][0:2]-leftEye_rightEye[0][0:2] # vector from eyeImg 0 to 1
                positionOfZeroWithinOne = self.pupilspacingrunningavg-imageZeroToOneVector; # the extra distance that wasn't covered by the bounding boxes should be applied as an offset when multiplying images.
                ksize = 5 #kernel size = x width and y height of the filter
                sigma = 2
                for i,centerEstimate in enumerate(pupilCenterEstimates):
                    pupilCenterEstimates[i] = cv2.GaussianBlur(pupilCenterEstimates[i], (ksize,ksize), sigma, borderType=cv2.BORDER_REPLICATE)
                jointPupilProb = self.multiply_prob_images(pupilCenterEstimates[1], pupilCenterEstimates[0], positionOfZeroWithinOne[::-1], 0) # the [::-1] reverse the order, so it's YX instead of the XY that these vectors are in
                ##            debugImg(jointPupilProb)
                maxInd = jointPupilProb.argmax()
                ##            cv2.imwrite( "eye0.png", pupilCenterEstimates[0]/pupilCenterEstimates[0].max()*255) #write probability images for our report
                ##            cv2.imwrite( "eye1.png", pupilCenterEstimates[1]/pupilCenterEstimates[1].max()*255)
                ##            cv2.imwrite( "eyeJoint.png", jointPupilProb/jointPupilProb.max()*255)
                (pupilCy,pupilCx) = np.unravel_index(maxInd, jointPupilProb.shape) # coordinates in the eye 1 (camera-right eye) image
                pupilXYList[0]=pupilXYList[1]=(pupilCx + leftEye_rightEye[1][0],pupilCy + leftEye_rightEye[1][1]) #convert to absolute image coordinates


            useSURFReference = True
            if not useSURFReference: # this code assumes you have drawn a dark dot on your forehead. Should be drawn between the eyes, about the size of the iris.
                dotSearchBox = np.round( centeredBox(leftEye_rightEye[0], leftEye_rightEye[1], xDistBetweenEyes*.2, xDistBetweenEyes*.3, -xDistBetweenEyes*.09 ) ).astype('int')

                (refY,refX) = self.get_pupil_center(gray[dotSearchBox[1]:dotSearchBox[3], dotSearchBox[0]:dotSearchBox[2]])
                refXY = (refX+dotSearchBox[0],refY+dotSearchBox[1])
                if allowDebugDisplay:
                    cv2.rectangle(output, (dotSearchBox[0], dotSearchBox[1]), (dotSearchBox[2], dotSearchBox[3]), (128,0,128), 1)
                    cv2.circle(output, refXY, 2, (0,0,100),thickness=1) #BGR format
            else: # Adam's virtual reference point code. See paper for how it works.
                refXY = (0,0)
                #global warm, virtualpoint
                self.warm += 1
                if self.warm > 8:
                    #adam
                    face = faces[0]#expect the first one
                    faceImg = gray[face[1]:face[3], face[0]:face[2]]
                    cornerImg = gray[corner[1]:corner[3], corner[0]:corner[2]]
                    if self.virtualpoint == None: #we haven't set up the reference point yet
                        haystackKeypoints, haystackDescriptors = self.detector.detectAndCompute(gray, mask=None)
                        if len(haystackKeypoints) != 0:
                            betweenEyes = (np.array(self.feature_center_xy(leftEye_rightEye[0]))+np.array(self.feature_center_xy(leftEye_rightEye[1])))/2
                            self.virtualpoint = ClassyVirtualReferencePoint.ClassyVirtualReferencePoint(haystackKeypoints, haystackDescriptors, (betweenEyes[0], betweenEyes[1]), face, leftEye_rightEye[0], leftEye_rightEye[1])
                        else:
                            print ("begin fail")
                    else: #we've already created it
                        keypoints, descriptors = self.detector.detectAndCompute(gray, mask=None)
                        if drawKeypoints:
                            imgToDrawOn = output
                        else:
                            imgToDrawOn = None
                        if len(descriptors) != 0:
                            refXY  = self.virtualpoint.getReferencePoint(keypoints, descriptors, face, leftEye_rightEye[0], leftEye_rightEye[1], imgToDrawOn)
                # end of Adam's reference point code

            for i in range(len(pupilXYList)):
                pupilXYList[i] = ( pupilXYList[i][0]-refXY[0], pupilXYList[i][1]-refXY[1])
            pupilXYList = list(pupilXYList[0])+ list(pupilXYList[1]) #concatenate cam-left and cam-right coordinate tuples to make a single length 4 vector [x,y,x,y]

            if trackAverageOffset: # this frame's estimated offset will be a weighted average of the new measurement and the last frame's estimated offset
                #global OffsetRunningAvg
                if self.offset_running_avg is None:
                    self.offset_running_avg = np.array( [0,0])
                weightOnNew = .4; #Tuned parameter, must be >0 and <=1.0. Increase for faster response, decrease for better noise rejection.
                currentOffset = (np.array(pupilXYList[:2])+np.array(pupilXYList[2:]))/2
                self.offset_running_avg = (1.0-weightOnNew)*self.offset_running_avg + weightOnNew*currentOffset
                pupilXYList = self.offset_running_avg
                ##            import pdb; pdb.set_trace()
                if allowDebugDisplay:
                    cv2.line(output, (int(refXY[0]),int(refXY[1])), (int(refXY[0]+pupilXYList[0]), int(refXY[1]+pupilXYList[1])), (0,255,100))

            if allowDebugDisplay and self.show_main_image:
                # Double size
                cv2.imshow(self.WINDOW_NAME, cv2.resize(output,(0,0), fx=2,fy=2,interpolation=cv2.INTER_NEAREST) )
                # original size

            return tuple(pupilXYList) # if trackAverageOffset, it's length 2 and holds the average offset. Else, it's length 4 (old code)

        else: # no valid face was found
            if allowDebugDisplay:
                cv2.imshow(self.WINDOW_NAME, cv2.resize(output,(0,0), fx=2,fy=2,interpolation=cv2.INTER_NEAREST) )
            return None

    def detect(self, img, cascade, minimumFeatureSize=(20,20)):
        if cascade.empty():
            raise(Exception("There was a problem loading your Haar Cascade xml file."))
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2] #convert last coord from (width,height) to (maxX, maxY)
        return rects

    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    #def getLeftAndRightEyes(faces, eyes):
    def get_left_and_right_eyes(self, faces, eyes):
        #loop through detected faces. We'll do our processing on the first valid one.
        if len(eyes)==0:
            return ()
        for face in faces:
            for i in range(eyes.shape[0]):
                for j in range(i+1,eyes.shape[0]):
                    leftEye = eyes[i] #by left I mean camera left
                    rightEye = eyes[j]
                    #eyes are arrays of the form [minX, minY, maxX, maxY]
                    if (leftEye[0]+leftEye[2]) > (rightEye[0]+rightEye[2]): #leftCenter is > rightCenter
                        rightEye, leftEye = leftEye, rightEye #swap
                    if self.contains(leftEye,rightEye) or self.contains(rightEye, leftEye):#they overlap. One eye containing another is due to a double detection; ignore it
                        self.debug_print('rejecting double eye')
                        continue
                    if leftEye[3] < rightEye[1] or rightEye[3] < leftEye[1]:#top of one is below (>) bottom of the other. One is likely a mouth or something, not an eye.
                        self.debug_print('rejecting non-level eyes')
                        continue
                    #if leftEye.minY()>face.coordinates()[1] or rightEye.minY()>face.coordinates()[1]: #top of eyes in top 1/2 of face
                    #    continue;
                    if not (self.contains(face,leftEye) and self.contains(face,rightEye)):#face contains the eyes. This is our standard of humanity, so capture the face.
                        self.debug_print("face doesn't contain both eyes")
                        continue
                    return (leftEye, rightEye)

        return ()


    def contains(self, outerFeature, innerFeature):
        p = self.feature_center_xy(innerFeature)
        #eyes are arrays of the form [minX, minY, maxX, maxY]
        return p[0] > outerFeature[0] and p[0] < outerFeature[2] and p[1] > outerFeature[1] and p[1] < outerFeature[3]

    #def featureCenterXY(rect):
    def feature_center_xy(self, rect):
        #eyes are arrays of the form [minX, minY, maxX, maxY]
        return (.5*(rect[0]+rect[2]), .5*(rect[1]+rect[3]))

    #def debugPrint(self, s):
    def debug_print(self, s):
        if self.VERBOSE:
            print(s)

    #def getPupilCenter(gray, getRawProbabilityImage=False):
    def get_pupil_center(self, gray, getRawProbabilityImage=False):
        ##    (scleraY, scleraX) = np.unravel_index(gray.argmax(),gray.shape)
        ##    scleraColor = colors[scleraY,scleraX,:]
        ##    img[scleraX,scleraY] = (255,0,0)
        ##    img.colorDistance(skinColor[:]).save(disp)
        ##    img.edges().save(disp)
        ##    print skinColor, scleraColor
        gray = gray.astype('float32')
        if self.BLOWUP_FACTOR != 1:
            gray = cv2.resize(gray, (0,0), fx=self.BLOWUP_FACTOR, fy=self.BLOWUP_FACTOR, interpolation=cv2.INTER_LINEAR)

        IRIS_RADIUS = gray.shape[0]*.75/2 #conservative-large estimate of iris radius TODO: make this a tracked parameter--pass a prior-probability of radius based on last few iris detections. TUNABLE PARAMETER
        #debugImg(gray)
        dxn = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3) #optimization opportunity: blur the image once, then just subtract 2 pixels in x and 2 in y. Should be equivalent.
        dyn = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        magnitudeSquared = np.square(dxn)+np.square(dyn)

        # ########### Pupil finding
        magThreshold = magnitudeSquared.mean()*.6 #only retain high-magnitude gradients. <-- VITAL TUNABLE PARAMETER
                        # The value of this threshold is critical for good performance.
                        # todo: adjust this threshold using more images. Maybe should train our tuned parameters.
        # form a bool array, unrolled columnwise, which can index into the image.
        # we will only use gradients whose magnitude is above the threshold, and
        # (optionally) where the gradient direction meets characteristics such as being more horizontal than vertical.
        gradsTouse = (magnitudeSquared>magThreshold) & (np.abs(4*dxn)>np.abs(dyn))
        lengths = np.sqrt(magnitudeSquared[gradsTouse]) #this converts us to double format
        gradDX = np.divide(dxn[gradsTouse],lengths) #unrolled columnwise
        gradDY = np.divide(dyn[gradsTouse],lengths)
        #debugImg(gradsTouse*255)
        #ksize = 7 #kernel size = x width and y height of the filter
        #sigma = 4
        ##blurredGray = cv2.GaussianBlur(gray, (ksize,ksize), sigma, borderType=cv2.BORDER_REPLICATE)
        #debugImg(gray)
        #blurredGray = cv2.blur(gray, (ksize,ksize)) #x width and y height. TODO: try alternately growing and eroding black instead of blurring?
        #isDark = blurredGray < blurredGray.mean()
        isDark = gray< (gray.mean()*.8)  #<-- TUNABLE PARAMETER
        #global self.DILATIONKERNEL
        isDark = cv2.dilate(isDark.astype('uint8'), self.DILATIONKERNEL) #dilate so reflection goes dark too
        ##isDark = cv2.erode(isDark.astype('uint8'), dilationKernel)
        ##debugImg(isDark*255)
        gradXcoords =np.tile( np.arange(dxn.shape[1]), [dxn.shape[0], 1])[gradsTouse] # build arrays holding the original x,y position of each gradient in the list.
        gradYcoords =np.tile( np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[gradsTouse] # These lines are probably an optimization target for later.
        minXForPupil = 0 #int(dxn.shape[1]*.3)
        ##original method
        ##centers = np.array([[phi(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) if isDark[cy][cx] else 0 for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
        #histogram method
        centers = np.array([[self.phi_with_hist(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS) if isDark[cy][cx] else 0 for cx in range(minXForPupil,dxn.shape[1])] for cy in range(dxn.shape[0])]).astype('float32')
        # display outputs for debugging
        ##centers = np.array([[phiTest(cx,cy,gradDX,gradDY,gradXcoords,gradYcoords) for cx in range(dxn.shape[1])] for cy in range(dxn.shape[0])])
        ##debugImg(centers)
        maxInd = centers.argmax()
        (pupilCy,pupilCx) = np.unravel_index(maxInd, centers.shape)
        pupilCx += minXForPupil
        pupilCy /= self.BLOWUP_FACTOR
        pupilCx /= self.BLOWUP_FACTOR
        if self.WRITEEYEDEBUGIMAGES:
            global eyeCounter
            eyeCounter = (eyeCounter+1)%5 #write debug image every 5th frame
            if eyeCounter == 1:
                cv2.imwrite( "eyeGray.png", gray/gray.max()*255) #write probability images for our report
                cv2.imwrite( "eyeIsDark.png", isDark*255)
                cv2.imwrite( "eyeCenters.png", centers/centers.max()*255)
        if getRawProbabilityImage:
            return (pupilCy, pupilCx, centers)
        else:
            return (pupilCy, pupilCx)

    #phiWithHist
    def phi_with_hist(self, cx,cy,gradDX,gradDY,gradXcoords,gradYcoords, IRIS_RADIUS):
        vecx = gradXcoords-cx
        vecy = gradYcoords-cy
        lengthsSquared = np.square(vecx)+np.square(vecy)
        # bin the distances between 1 and IRIS_RADIUS. We'll discard all others.
        binWidth = 1 #TODO: account for webcam resolution. Also, maybe have it transform ellipses to circles when on the sides? (hard)
        numBins =  int(np.ceil((IRIS_RADIUS-1)/binWidth))
        bins = [(1+binWidth*index)**2 for index in range(numBins+1)] #express bin edges in terms of length squared
        hist = np.histogram(lengthsSquared, bins)[0]
        maxBin = hist.argmax()
        slop = binWidth
        valid = (lengthsSquared > max(1,bins[maxBin]-slop)) &  (lengthsSquared < bins[maxBin+1]+slop) #use only points near the histogram distance
        dotProd = np.multiply(vecx,gradDX)+np.multiply(vecy,gradDY)
        valid = valid & (dotProd > 0) # only use vectors in the same direction (i.e. the dark-to-light transition direction is away from us. The good gradients look like that.)
        dotProd = np.square(dotProd[valid]) # dot products squared
        dotProd = np.divide(dotProd,lengthsSquared[valid]) #make normalized squared dot products
        #dotProd = dotProd[dotProd > .9] #only count dot products that are really close
        dotProd = np.square(dotProd) # squaring puts an even higher weight on values close to 1
        return np.sum(dotProd) # this is equivalent to normalizing vecx and vecy, because it takes dotProduct^2 / length^2

    # multiplies newProb and priorToMultiply
    # YXoffsetOfSecondWithinFirst - priorToMultiply will be shifted by this amount in space
    # defaultPriorValue - if not all of newProb is covered by priorToMultiply, this scalar goes in the uncovered areas.
    #multiplyProbImages
    def multiply_prob_images(self, newProb, priorToMultiply, YXoffsetOfSecondWithinFirst, defaultPriorValue):
        if np.any(YXoffsetOfSecondWithinFirst > newProb.shape) or np.any(-YXoffsetOfSecondWithinFirst > priorToMultiply.shape):
            print ("multiplyProbImages aborting - zero overlap. Offset and matrices:")
            print (YXoffsetOfSecondWithinFirst)
            print (newProb.shape)
            print (priorToMultiply.shape)
            return newProb*defaultPriorValue
        prior = np.ones(newProb.shape)*defaultPriorValue # Most of this will get overwritten. For areas that won't be, with fill with default value.
        #offsets
        startPrior = [0,0]
        endPrior = [0,0]
        startNew = [0,0]
        endNew = [0,0]
        for i in range(2):
            #offset=0
            # NOT THIS: x[1:2][1:2]
            # THIS: x[1:2,1:2]
            offset = int(round(YXoffsetOfSecondWithinFirst[i])) # how much to offset 'prior' within 'newProb', for the current dimension
            print (offset)
            if offset >= 0: # prior goes right of 'newProb', in the world. So prior will be copied into newProb at a positive offset
                startPrior[i] = 0 #index within prior
                endPrior[i] = min(priorToMultiply.shape[i],newProb.shape[i]-offset) #how much of prior to copy
                startNew[i]=offset
                endNew[i]=offset+endPrior[i]
            else: # prior goes left of 'newProb', in the world.
                startPrior[i] = -offset
                endPrior[i] = min(priorToMultiply.shape[i], startPrior[i]+newProb.shape[i])
                startNew[i]=0
                endNew[i]=endPrior[i]-startPrior[i]
        prior[startNew[0]:endNew[0],startNew[1]:endNew[1]] = priorToMultiply[startPrior[0]:endPrior[0],startPrior[1]:endPrior[1]]
        #prior[1:10,1:10] = priorToMultiply[1:10,1:10]
        #now, prior holds the portion of priorToMultiply which overlapped newProb.
        return newProb * prior

    def make_detector(self):
        hessianThreshold = 500
        nOctaves = 4
        nOctaveLayers = 2
        extended = True
        upright = True
        self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)


class Training(CheckCamera):

    RANSAC_MIN_INLIERS = 7
    readSuccessful = False

    def __init__(self, *args, **kwargs):
        from . import pygamestuff
        self.make_detector()
        crosshair = pygamestuff.Crosshair([7, 2], quadratic = False)
        vc = cv2.VideoCapture(0) # Initialize the default camera
        if vc.isOpened(): # try to get the first frame
            (self.readSuccessful, frame) = vc.read()
        else:
            raise(Exception("failed to open camera."))
        MAX_SAMPLES_TO_RECORD = 999999
        recordedEvents=0 #numero de fijaciones
        HT = None
        XYPupilFrame.objects.all().delete()
        try:
            coords = []
            points = 0
            clicks = 0
            while self.readSuccessful and recordedEvents < MAX_SAMPLES_TO_RECORD and not crosshair.userWantsToQuit:
                points += 1
                pupilOffsetXYList = self.get_offset(frame, allowDebugDisplay=False)
                if pupilOffsetXYList is not None: #si se obtienen los dos ojos, espera un click
                    if crosshair.pollForClick(): #si hace click se agregan los puntos a la calibracion
                        clicks += 1
                        #print('clicks '+ str(clicks))
                        crosshair.clearEvents()
                        #print( (xOffset,yOffset) )
                        #do learning here, to relate xOffset and yOffset to screenX,screenY
                        crosshair.record(pupilOffsetXYList)
                        print(pupilOffsetXYList)

                        xyframemodel = XYPupilFrame()
                        xyframemodel.x = pupilOffsetXYList[0]
                        xyframemodel.y = pupilOffsetXYList[1]
                        xyframemodel.save()

                        print ("recorded something")
                        crosshair.remove()
                        recordedEvents += 1
                        if recordedEvents > self.RANSAC_MIN_INLIERS:
                            ##HT = fitTransformation(np.array(crosshair.result))
                            resultXYpxpy =np.array(crosshair.result)
                            features =self.get_features(resultXYpxpy[:,:-2])
                            featuresAndLabels = np.concatenate( (features, resultXYpxpy[:,-2:] ) , axis=1)
                            HT = self.RANSACFitTransformation(featuresAndLabels)
                            print (HT)
                    if HT is not None: # dibujar el circulo estimando la mirada
                        #print('ya empieza la estimacion')
                        #print(messagebox.askyesnocancel(message="Comenzará la calibración", title="Título"))

                        fixations = 0
                        currentFeatures = self.get_features( np.array( (pupilOffsetXYList[0], pupilOffsetXYList[1]) ))
                        gazeCoords = currentFeatures.dot(HT)
                        crosshair.drawCrossAt((gazeCoords[0,0], gazeCoords[0,1]))
                        print(gazeCoords[0,0], gazeCoords[0,1])
                        coords.append({
                        'fixation_number': fixations, 'x': gazeCoords[0,0],'y': gazeCoords[0,1] #las fijaciones son los puntos que detecta la aplicacion que un usuario mira
                    })
                        fixations += 1
                self.readSuccessful, frame = vc.read()
        
        
            crosshair.write() #writes data to a csv for MATLAB
            crosshair.close()
            resultXYpxpy = np.array(crosshair.result)
        finally:
            vc.release() #close the camera
            #self.make_model(coords)

    '''
    def main_for_training():
        import pygamestuff
        crosshair = pygamestuff.Crosshair([7, 2], quadratic = False)
        vc = cv2.VideoCapture(0) # Initialize the default camera
        if vc.isOpened(): # try to get the first frame
            (self.readSuccessful, frame) = vc.read()
        else:
            raise(Exception("failed to open camera."))
            return

        MAX_SAMPLES_TO_RECORD = 999999
        recordedEvents=0
        HT = None
        try:
            while self.readSuccessful and recordedEvents < MAX_SAMPLES_TO_RECORD and not crosshair.userWantsToQuit:
                pupilOffsetXYList = getOffset(frame, allowDebugDisplay=False)
                if pupilOffsetXYList is not None: #If we got eyes, check for a click. Else, wait until we do.
                    if crosshair.pollForClick():
                        crosshair.clearEvents()
                        #print( (xOffset,yOffset) )
                        #do learning here, to relate xOffset and yOffset to screenX,screenY
                        crosshair.record(pupilOffsetXYList)
                        print "recorded something"
                        crosshair.remove()
                        recordedEvents += 1
                        if recordedEvents > RANSAC_MIN_INLIERS:
        ##                    HT = fitTransformation(np.array(crosshair.result))
                            resultXYpxpy =np.array(crosshair.result)
                            features = getFeatures(resultXYpxpy[:,:-2])
                            featuresAndLabels = np.concatenate( (features, resultXYpxpy[:,-2:] ) , axis=1)
                            HT = RANSACFitTransformation(featuresAndLabels)
                            print HT
                    if HT is not None: # draw predicted eye position
                        currentFeatures =getFeatures( np.array( (pupilOffsetXYList[0], pupilOffsetXYList[1]) ))
                        gazeCoords = currentFeatures.dot(HT)
                        crosshair.drawCrossAt( (gazeCoords[0,0], gazeCoords[0,1]) )
                self.readSuccessful, frame = vc.read()
        
            print "writing"
            crosshair.write() #writes data to a csv for MATLAB
            crosshair.close()
            print "HT:\n"
            print HT
            resultXYpxpy =np.array(crosshair.result)
            print "eyeData:\n"
            print getFeatures(resultXYpxpy[:,:-2])
            print "resultXYpxpy:\n"
            print resultXYpxpy[:,-2:]
        finally:
            vc.release() #close the camera
    '''

    def make_model(self, coords, aois):
        """
        El modelo es la unión de las zonas que se etiquetaron y las fijaciones

        """

        segmentation = aois
        functions = {}
        relations = []
        print(coords)
        print(segmentation)
        for index_fixation, row in enumerate(coords): #se verificar cuáles fijaciones se encuentran dentro de las zonas que se definieron y si es así se agregan al modelo
            relations.append((index_fixation, index_fixation + 1))
            for index_segment, segment in enumerate(segmentation):
                if (float(row['x']) >= float(segment['x0']) and float(row['x']) <= float(segment['x1'])):
                    if (float(row['y']) >= float(segment['y0']) and float(row['y']) <= float(segment['y1'])):
                        # print ('fixation index', index_fixation, 'segment index', index_segment, 'match')
                        functions.update({index_fixation: [segment['aoi']]})
                        break
                    else:
                        functions.update({index_fixation: ['undefined']})
                else:
                    functions.update({index_fixation: ['undefined']})
        relations.append((len(relations), len(relations)))
        return relations, functions
        '''
        def getResult(relations,functions):
            """
            Una vez que se tiene el modelo, se le pide al usuario la fórmula que es la que se va a ocupar para verificar si se cumple o no

            """

            phi = theformula.get()
            K = Kripke(R=relations, L=functions)
            print(K)
            print(modelcheck(K, phi))

            result = modelcheck(K, phi)
            return result
        '''
            # w_result = tk.Tk()
            #
            # w_result.title("Result")
            # label_one = tk.Label(w_result, text="Formula")
            # label_one.grid(row=0, column=0)
            # label_two = tk.Label(w_result, text=phi)
            # label_two.grid(row=0, column=1)

            #label_three = tk.Label(formula, text="Result")
            #label_three.grid(row=1, column=0)
            #label_three = tk.Label(formula, text=result)
            #label_three.grid(row=1, column=1)

        #img = cv2.imread('lenguajes.jpg')
        #for index_fixation, row in enumerate(coords):
        #    print(int(row['x']), int(row['y']))
        #    # cv2.circle(img, (int(row['x']), int(row['y'])), 15, (0, 0, 255), -1)
        '''
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(row['x']), int(row['y']))
            fontScale = 1
            fontColor = (0, 0, 255)
            lineType = 2

            cv2.putText(img, str(index_fixation),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            # img[int(row['y']), int(row['x'])] = [0, 0, 255]
        
        cv2.imshow('image', img)
        cv2.imwrite("scanpath.png", img) #save the last-displayed image to file, for our report
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # formula = tk.Tk()
        #
        # formula.title("Formula")
        # labelone = tk.Label(formula, text="Formula")
        # labelone.grid(row=0, column=0)
        # name = tk.StringVar()
        #
        # theformula = tk.Entry(formula, textvariable=name)
        # theformula.grid(row=0, column=1)
        # btn = tk.Button(formula, text="Model check", command= lambda: getResult(relations,functions))
        # btn.grid(row=0, column=3)


    def getResult(self, relations,functions, phi=""):
        """
        Una vez que se tiene el modelo, se le pide al usuario la fórmula que es la que se va a ocupar para verificar si se cumple o no

        """

        K = Kripke(R=relations, L=functions)
        print(K)
        print(modelcheck(K, phi))

        result = modelcheck(K, phi)
        return result

    def get_features(self, XYOffsets, quadratic = True):
        """

        esta funcion se encarga de renderar y de hacer tal cosa,

        """
    ##    print XYOffsets
        if len(XYOffsets.shape)==1:
            numRows=1
            XYOffsets.shape = (numRows,XYOffsets.shape[0])
        else:
            numRows =XYOffsets.shape[0]
        numCols = XYOffsets.shape[1]

        data = np.concatenate( (XYOffsets, np.ones( (XYOffsets.shape[0],1)) ) , axis=1) # [x,y,1]
        if quadratic:
            squaredFeatures = np.square(XYOffsets)
            squaredFeatures.shape = (numRows,numCols)
            xy = XYOffsets[:,0]*XYOffsets[:,1]
            xy.shape = (numRows,1)
    ##        print(xy.shape)

            data = np.concatenate( (data,squaredFeatures, xy ) , axis=1) # [x,y,1,x^2,y^2,xy]
        return data

    def RANSACFitTransformation(self, OffsetsAndPixels):
        numInputCols = OffsetsAndPixels.shape[1]-2
        data = np.concatenate( (OffsetsAndPixels[:,0:numInputCols], OffsetsAndPixels[:,numInputCols:] ) , axis=1)

        model = LinearLeastSquaresModel(range(numInputCols), (numInputCols,numInputCols+1))
        minSeedSize = 5
        iterations = 800
        maxInlierError = 240 #**2
        HT = ransac.ransac(data, model, minSeedSize, iterations, maxInlierError, self.RANSAC_MIN_INLIERS)
        return HT


class MakeExperiment(Training):

    start_tk = False
    readSuccessful = False
    app_has_destroy = False

    def __init__(self, image_root, experiment):
        from . import pygamestuff
        self.make_detector()
        self.image_root = image_root
        self.experiment = experiment
        crosshair = pygamestuff.Crosshair([7, 2], quadratic = False)
        vc = cv2.VideoCapture(0) # Initialize the default camera
        if vc.isOpened(): # try to get the first frame
            (self.readSuccessful, frame) = vc.read()
        else:
            raise(Exception("failed to open camera."))
        MAX_SAMPLES_TO_RECORD = 999999
        recordedEvents=0 #numero de fijaciones
        HT = None
        XYPupilFrame.objects.all().delete()
        coords = []
        points = 0
        clicks = 0
        try:
            while (self.readSuccessful and recordedEvents < MAX_SAMPLES_TO_RECORD and not crosshair.userWantsToQuit) and not self.app_has_destroy:
                points += 1
                pupilOffsetXYList = self.get_offset(frame, allowDebugDisplay=False)
                if pupilOffsetXYList is not None: #si se obtienen los dos ojos, espera un click
                    if crosshair.pollForClick(): #si hace click se agregan los puntos a la calibracion
                        clicks += 1
                        #print('clicks '+ str(clicks))
                        crosshair.clearEvents()
                        #print( (xOffset,yOffset) )
                        #do learning here, to relate xOffset and yOffset to screenX,screenY
                        crosshair.record(pupilOffsetXYList)
                        print(pupilOffsetXYList)

                        xyframemodel = XYPupilFrame()
                        xyframemodel.x = pupilOffsetXYList[0]
                        xyframemodel.y = pupilOffsetXYList[1]
                        xyframemodel.save()

                        print ("recorded something")
                        crosshair.remove()
                        recordedEvents += 1
                        if recordedEvents > self.RANSAC_MIN_INLIERS:
                            ##HT = fitTransformation(np.array(crosshair.result))
                            
                            resultXYpxpy =np.array(crosshair.result)
                            features =self.get_features(resultXYpxpy[:,:-2])
                            featuresAndLabels = np.concatenate( (features, resultXYpxpy[:,-2:] ) , axis=1)
                            HT = self.RANSACFitTransformation(featuresAndLabels)
                            print (HT)
                    if HT is not None: # dibujar el circulo estimando la mirada
                        #print('ya empieza la estimacion')
                        #print(messagebox.askyesnocancel(message="Comenzará la calibración", title="Título"))
                        show_image_thread = threading.Thread(target=self.show_image)
                        if not self.start_tk:
                            show_image_thread.start()
                            self.start_tk = True
                        fixations = 0
                        currentFeatures = self.get_features( np.array( (pupilOffsetXYList[0], pupilOffsetXYList[1]) ))
                        gazeCoords = currentFeatures.dot(HT)
                        crosshair.drawCrossAt((gazeCoords[0,0], gazeCoords[0,1]))
                        print(gazeCoords[0,0], gazeCoords[0,1])
                        coords.append({
                            'fixation_number': fixations, 
                            'x': gazeCoords[0,0],
                            'y': gazeCoords[0,1] #las fijaciones son los puntos que detecta la aplicacion que un usuario mira
                        })
                        fixations += 1
                    print("coords : ", coords)
                self.readSuccessful, frame = vc.read()        
        
            crosshair.write() #writes data to a csv for MATLAB
            crosshair.close()
            resultXYpxpy = np.array(crosshair.result)
        finally:
            vc.release() #close the camera
            #relations, functions = self.make_model(coords)
            #result = self.getResult(relations, functions)
            for coor in coords:
                experiment_point = ExperimentPoint()
                experiment_point.fixation_number = coor.get('fixation_number')
                experiment_point.x = coor.get('x')
                experiment_point.y = coor.get('y')
                experiment_point.experiment = self.experiment
                experiment_point.save()
 
    def show_image(self):
        ShowImage(self.image_root, self)

class GenerateResults(MakeExperiment):

    def __init__(self, experiment, formula):
        self.experiment = experiment
        self.phi = formula
        self.phi = eval(self.phi)
        #self.generate_results()

    def generate_result(self):
        experiment = self.experiment
        coords = [{'fixation_number': point.fixation_number, 'x': float(point.x), 'y': float(point.y)} for point in experiment.points.all() ]
        aois = [{"aoi": aoi.name,
                "x0": float(aoi.x0),
                "x1": float(aoi.x1), 
                "y0": float(aoi.y0),
                "y1": float(aoi.y1)} for aoi in experiment.image.rectangles.all()]
        relations, functions = self.make_model(coords, aois)        
        result = self.getResult(relations,functions, phi=self.phi)
        return relations, functions, result