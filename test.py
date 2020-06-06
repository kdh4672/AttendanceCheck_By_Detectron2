import time
from detectron2.config import get_cfg
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import numpy as np
from PIL import ImageFont, ImageDraw, Image # 한글 폰트 사용하기 위해
from Acheck_TR import Acheck_TR

cfg = get_cfg()
yaml = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
weight = './output/model_final_cascade_with_hair/mask_rcnn_R_101_C4_3x/model_0029999.pth'
cfg.merge_from_file(
    yaml
)
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
## (load pretrained weights)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (Kong, lee, Huh)
register_coco_instances("Acheck", {}, "./Acheck_hair.json", "./img_hair")
MetadataCatalog.get("Acheck").thing_classes = ["Kong", "Lee" , "Huh"]
Acheck_metadata = MetadataCatalog.get("Acheck")
cfg.MODEL.WEIGHTS = weight
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
cfg.DATASETS.TEST = ("Acheck", )
predictor = DefaultPredictor(cfg)

def image_test(start,end):
  for i in range(start,end):
    k = "./test_images/{}.jpg".format(i)
    im = cv2.imread(k)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                      metadata=Acheck_metadata,
                      scale=0.8,
                      instance_mode=ColorMode.IMAGE_BW)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("d",v.get_image()[:, :, ::-1])
    cv2.waitKey()

class Person(object):
  def __init__(self,SN,major,name):
    self.SN= SN
    self.major = major
    self.name = name
    self.appear = 0
    self.percent = 0
    self.result = 0
person1 = Person(20151739,'기계공학과','공대현(Kong)')
person2 = Person(20151476,'전자공학과','이원석(Lee)')
person3 =Person(20141713,'화생공학과','허  찬(Huh)')
person4 = Person(20161739, '방송연예학', '사  나(Sana)')
person_list = [person1, person2, person3, person4]

anum=[ 0 , 0 , 0 ,0]
aboard = {0:'Kong',1:'Lee',2:'Huh',3:'Sana'}
b , g , r , a= 0 , 255 , 0 , 0
fontpath="./fonts/gulim.ttf"
font = ImageFont.truetype( fontpath , 20 )
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

def video_test(video_file,frame_term):        #frame_term must be integer
    stream = cv2.VideoCapture(video_file)
    frame = -1
    start = time.time()

    while True :
      frame = frame + 1
      rr , im=stream.read()
      if rr == False:
        break
      if frame%frame_term != 0:
        pass
      else:
        #im = cv2.flip(im , 0) # 상하반전
        outputs=predictor( im )
        v = Visualizer(im[:, :, ::-1],
                        metadata=Acheck_metadata,
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW)


        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]
        #####################################################################
        classes=outputs[ 'instances' ].__dict__.get( '_fields' ).get( 'pred_classes' )
        classes_list=list( set( classes.tolist() ) )
        l=len( classes_list )

        for i in range(l):
          anum[ classes_list[ i ] ]=anum[ classes_list[ i ] ] + 1

        for i in range(len( anum )) :
          if anum[ i ] :
            img_pil=Image.fromarray( img )
            draw=ImageDraw.Draw( img_pil )
            str = "{}:{} frame".format(aboard.get(i),anum[i])
            draw.text( (60 , 30+i*30) , str , font=font , fill=(b , g , r , a) )
            img=np.array( img_pil )
        #####################################################################
        if frame/frame_term == 8 :
          print( "지각검사 체크" )
          print('when frame is 8 :',anum)
          for i in range(len(anum)):
            if anum[i] <= 1:
              person_list[i].result = '지각(late)'
        if frame/frame_term == 12 :
          print( "결석검사 체크" )
          print( 'when frame is 12 :' , anum )
          for i in range(len(anum)):
            if person_list[i].result != '지각(late)':
              continue
            if anum[i] <= 1 :
              person_list[i].result = '결석(absence)'

        #####################################################################
        cv2.imshow("video_test,frame term:{}".format(frame_term), img)
      if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) :
        break
    total_time = time.time()-start
    frame = frame/frame_term
    atime=[ 0 , 0 , 0 ,0]
    Acheck_TR(anum,atime,person_list,frame,total_time)


def ipcam_test():
  stream = cv2.VideoCapture(0)
  start=time.time()
  frame = 0
  while True :
    frame += 1
    rr , im=stream.read()
    outputs=predictor( im )
    v=Visualizer( im[ : , : , : :-1 ] , metadata=Acheck_metadata , scale=0.8 , instance_mode=ColorMode.IMAGE_BW )
    v=v.draw_instance_predictions( outputs[ "instances" ].to( "cpu" ) )
    img=v.get_image()[ : , : , : :-1 ]
    #####################################################################
    classes=outputs[ 'instances' ].__dict__.get( '_fields' ).get( 'pred_classes' )
    classes_list=list( set( classes.tolist() ) )
    l=len( classes_list )
    for i in range( l ) :
      anum[ classes_list[ i ] ]=anum[ classes_list[ i ] ] + 1
    for i in range( len( anum ) ) :
      if anum[ i ] :
        img_pil=Image.fromarray( img )
        draw=ImageDraw.Draw( img_pil )
        str="{}:{} frame".format( aboard.get( i ) , anum[ i ] )
        draw.text( (60 , 30+i*30) , str , font=font , fill=(b , g , r , a) )
        img=np.array( img_pil )

    #####################################################################
    if frame  == 8 :
      print( "지각검사 체크" )
      print( 'when frame is 8 :' , anum )
      for i in range( len( anum ) ) :
        if anum[ i ] <= 1 :
          person_list[ i ].result='지각(late)'
    if frame  == 12 :
      print( "결석검사 체크" )
      print( 'when frame is 12 :' , anum )
      for i in range( len( anum ) ) :
        if person_list[ i ].result != '지각(late)' :
          continue
        if anum[ i ] <= 3 :
          person_list[ i ].result='결석(absence)'

    #####################################################################
    cv2.imshow( "ipcam_test" , img )

    if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) :
      break
  total_time = time.time()-start
  frame = frame
  atime=[ 0 , 0 , 0 ,0]
  Acheck_TR(anum,atime,person_list,frame,total_time)


def mAPtest(yaml,weight):
  from detectron2.engine import DefaultTrainer
  from detectron2.config import get_cfg
  cfg=get_cfg()
  cfg.merge_from_file( yaml)
  cfg.DATALOADER.NUM_WORKERS=2
  # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
  cfg.MODEL.ROI_HEADS.NUM_CLASSES=3  # 3 classes (Kong, lee, Huh)

  from detectron2.data.datasets import register_coco_instances
  register_coco_instances( "Acheck_test" , { } , "./Acheck_hair_test.json" , "./img_hair_test" )

  from detectron2.data import MetadataCatalog
  MetadataCatalog.get( "Acheck_test" ).thing_classes=[ "Kong" , "Lee" , "Huh" ]
  Acheck_metadata=MetadataCatalog.get( "Acheck_test" )
  from detectron2.data import DatasetCatalog
  dataset_dicts=DatasetCatalog.get( "Acheck_test" )
  cfg.DATASETS.TRAIN=("Acheck_test" ,)

  from detectron2.engine import DefaultPredictor

  cfg.MODEL.WEIGHTS= weight
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.8  # set the testing threshold for this model
  cfg.DATASETS.TEST=("Acheck_test" ,)
  predictor=DefaultPredictor( cfg )
  trainer=DefaultTrainer( cfg )
  trainer.resume_or_load( resume=False )

  from detectron2.evaluation import COCOEvaluator , inference_on_dataset
  from detectron2.data import build_detection_test_loader
  evaluator=COCOEvaluator( "Acheck_test" , cfg , False , "./output/" )
  val_loader=build_detection_test_loader( cfg , "Acheck_test" )
  inference_on_dataset( trainer.model , val_loader , evaluator )

###################################################################################################
#ipcam_test()
video_test('./test_videos/13.mp4',10)
#image_test(1,20)
#mAPtest(yaml,weight)