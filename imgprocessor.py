#python3 

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

def capture():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0


    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            return None
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            frame = None
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break
    cam.release()

    cv2.destroyAllWindows()
    return frame

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def houghlines(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/4,300)
    counter = 0
    for l in lines:
      for rho,theta in l:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        counter = counter + 1

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    print(lines)
    #show(img)
    print ("There were %d lines"%(counter))


def put_val_in_list_with_avg_merging(thelist, newval, max_marge):

    # Returns a list of (value,weight) with the added integer value. If the added value is near an other existing value, then
    # both are replaced with the average (taking previous weight into account)

    rlist = []
    merged = False
    for aval,weight in thelist:
        if abs(aval-newval) > max_marge:
            rlist.append((aval,weight))
        else:
            newweight = weight + 1
            rlist.append( (int( ( aval * weight + newval ) / newweight ), newweight) )
            merged = True
    if not merged:
        rlist.append((newval,1))
    return rlist

def keep_periodic_values(list):
    list.sort()
    values_with_intervals=[]
    weighted_intervals = []
    for l in range (0,len(list)):
        if l>0:
            interval_before = list[l] - list[l-1]
            weighted_intervals = put_val_in_list_with_avg_merging(weighted_intervals, interval_before, 5)
        else:
            interval_before = -1;
        if l < len(list) - 1:
            interval_after = list[l+1] - list[l]
            weighted_intervals = put_val_in_list_with_avg_merging(weighted_intervals, interval_after, 5)
        else:
            interval_after = -1;
        values_with_intervals.append((list[l], interval_before, interval_after))
        
    weighted_intervals.sort(key=(lambda vw: vw[1]), reverse = True)
    print("weighted intervals : %s"%(weighted_intervals))

    period = weighted_intervals[0][0];


    return [ v  for (v,ib,ia) in values_with_intervals if ( abs(ib-period) < 5 or abs(ia-period) < 5 )]


def lcfind(origimg):

    dimensions = origimg.shape
     
    # height, width, number of channels in image
    height = origimg.shape[0]
    width = origimg.shape[1]
    channels = origimg.shape[2]
 
    print("Height = %d   Width = %d"%(width,height))

    gray = cv2.cvtColor(origimg,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/4,300)
    cols=[]
    rows=[]
    for l in lines:
      for rho,theta in l:
        if abs(theta) > np.pi/10:
            # Here we have a vertical line
            y = np.sin(theta) * rho
            rows = put_val_in_list_with_avg_merging(rows, y, 5)
        else:
            x = np.cos(theta) * rho
            cols = put_val_in_list_with_avg_merging(cols, x, 5)



    rows = keep_periodic_values([v for v,w in rows])
    cols = keep_periodic_values([v for v,w in cols])       

    gridperiod = rows[1] - rows [0]

    xmin = min(cols)
    xmax = max(cols)
    ymin = min(rows)
    ymax = max(rows)
    img = origimg
    for x in cols:
        print ("Vertical line at x=%d"%(x))
        cv2.line(img,(x,ymin),(x,ymax),(0,0,255),2)
    for y in rows:
        print ("Horizontal line at y=%d"%(y))
        cv2.line(img,(xmin,y),(xmax,y),(0,0,255),2)
    
    
    print ("There were %d vertical lines and %d horizontal  lines"%(len(cols), len(rows)))

    gs = get_grayscale(img)
    thim = thresholding(gs)


    #show (img)

    imgs =  [ [ thim[rows[yl]+2:rows[yl+1]-2, cols[xl]+2:cols[xl+1]-2]  for yl in range(0,len(rows)-1)] for xl in range (0,len(cols)-1) ] 


    #digit_img = imgs[2][1] # 3eme ligne en partant du haut, 2ème colonne en partant de la gauche
    #show (digit_img) # 3eme ligne en partant du haut, 2ème colonne en partant de la gauche
    
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    #print(pytesseract.image_to_string(digit_img, config=custom_config))
    
    nbcols = len(cols) -1
    nbrows = len(rows) - 1

    digits=[]
    for img_col in imgs:
        col_digits = [ pytesseract.image_to_string(digit_img, config=custom_config) for digit_img in img_col]
        digits.append(col_digits)


    print (digits[2][1]) # 3eme ligne en partant du haut, 2ème colonne en partant de la gauche

    show(img)

    digits[3][1]="X"
    digits[2][2]="X"
    digits[0][2]="X"
    digits[1][1]="X"
    digits[2][0]="."

    for xl in range(0,nbcols):
        for yl in range(0,nbrows):
            if digits[xl][yl] == "X":
                cv2.rectangle(img,(cols[xl],rows[yl]),(cols[xl+1],rows[yl+1]),(0,0,0),-1)
            elif digits[xl][yl] == ".":
                cv2.circle(img,(int((cols[xl]+cols[xl+1])/2),int((rows[yl]+rows[yl+1])/2)), int(gridperiod/10),(0,0,0),-1)

                
                #print("x=%d y=%d val=%s"%(xl,yl,digits[xl][yl]))

    #    print(digits)
    show(img)

    for yl in range(0,nbrows):
        rowstring = ""
        for xl in range(0,nbcols):
            rowstring = rowstring + " " + digits[xl][yl]
        print (rowstring)
    


    #d = pytesseract.image_to_data(digit_img, output_type=Output.DICT)
    #print(d)
#
 #   show(img)
  

   
def HoughlinesP(img):
    gray = get_grayscale(img)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)print ("Horizontal line at y=%d"%(y))
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP( edges, 10, np.pi/180, 100,100)
    for l in lines:
      for x1,y1,x2,y2 in l:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

img = cv2.imread('simple.jpg')
#img = capture()
lcfind(img)

#houghlines(img)


#img = thresholding(img)

#d = pytesseract.image_to_data(img, output_type=Output.DICT)
#print(d)
#print(d.keys())

# show(img)