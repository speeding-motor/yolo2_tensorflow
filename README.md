# yolo2_tensorflow

this project is the implements of the yolo2 paper from scrath, i will record the troble and complete detail here


Troble:
1、when i use :
	priors = tf.reshape(Anchors, [1, 1, 1, anchor_num, 2])
        pred_wh = tf.exp(pred[..., 3:5]) * priors
	
   it always occurred NAN error, because pred[..., 3:5] over than hundred,


Implement Detail:
1、pass the result of the kemeans ,wo found that k=5 is the best anchor box classsify for voc2012 dataset
2、
