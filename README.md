#   Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs, ICCV 2021

**[ Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs](https://openaccess.thecvf.com/content/ICCV2021/html/Islam_Global_Pooling_More_Than_Meets_the_Eye_Position_Information_Is_ICCV_2021_paper.html)**
<br>
**[Md Amirul Islam*](https://www.cs.ryerson.ca/~amirul/)**, **[Matthew Kowal*](https://mkowal2.github.io/)**, **[Sen Jia](https://scholar.google.com/citations?user=WOsy1foAAAAJ&hl=en)**, **[Konstantinos G. Derpanis](https://www.cs.ryerson.ca/~kosta/)**, **[Neil Bruce](http://socs.uoguelph.ca/~brucen/)** 

<br>

#  Channel-wise Position Encoding

1. Train and Test GAPNet for location classification or image recognition using the following commands:

            cd channel-wise-position-encoding/
            python trainval_gapnet.py 
            python test_gapnet.py 
            
2. Train and Test PermuteNet for location classification or image recognition using the following commands:

            cd channel-wise-position-encoding/
            python trainval_permutenet.py 
            python test_permutenet.py 
 
 
 #  Learning Translation Invariant Representation
 Code coming soon!
 

 #  Targeting Position-Encoding Channels
 
   Identify and Rank the position encoding channels followed by targeting the ranked channels using the following commands:

            cd position_attack/
            bash run_rank_target_neurons.sh
  
   Please download the DeepLabv3-ResNet50 model trained on Cityscapes from [Dropbox](https://www.dropbox.com/s/n6zr9snkx6qd5ms/zero_padding_best_deeplabv3_resnet50_cityscapes_os16.pth?dl=0) and put it under .checkpoints/
  
  Download the cityscapes dataset and change the dataset root path accordingly!
 
<br>

# BibTeX
If you find this repository useful, please consider giving a star :star: and citation :t-rex:


      @InProceedings{islam2021global,
       title={Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs},
       author={Islam, Md Amirul and Kowal, Matthew and Jia, Sen and Derpanis, Konstantinos G and Bruce, Neil},
       booktitle={International Conference on Computer Vision},
       year={2021}
     }

