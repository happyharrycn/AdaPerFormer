###
 # @Author: Owen Young
 # @Date: 2022-04-19 17:32:19
 # @LastEditTime: 2022-08-24 15:27:35
 # @LastEditors: Owen Young
 # @Description: 
 # @FilePath: /AdaPerFormer/src/test_thumos.sh
### 

CUDA_VISIBLE_DEVICES=0 python3 /eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_0421 ./logs/log_thumos_0421.log 2>&1 &