# yolo2_tensorflow

### this project is the implements of the yolo2 paper from scrath, i will record the troble and complete detail here


# Troble:

1. #### NAN , LOSS INCrease ERROR 

   when i use :	

   ```
   priors = tf.reshape(Anchors, [1, 1, 1, anchor_num, 2])
   pred_wh = tf.exp(pred[..., 3:5]) * priors
   ```

      it always occurred NAN error, because pred[..., 3:5] always over than hundred after 1~2 train epoch,

   And train loss is always been biger in training, i need to figure it out


# Implement Detail:

1. #### auto_kemeans: use kemeans to generate the anchor_box automaticlly

   
