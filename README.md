# FMD
Feature Map Distillation of Thin Nets for Low-resolution Object Recognition

  Intelligent video surveillance is an important computer vision application in natural environments. Since detected objects under surveillance are usually low-resolution and noisy, their accurate recognition represents a huge challenge. Knowledge distillation is an effective method to deal with it, but existing related work usually focuses on reducing the channel count of a student network, not feature map size. As a result, they cannot transfer “privilege information” hidden in feature maps of a wide and deep teacher network into a thin and shallow student one, leading to the latter’s poor performance. To address this issue, we propose a Feature Map Distillation (FMD) framework under which the feature map size of teacher and student networks is different. FMD consists of two main components: Feature Decoder Distillation (FDD) and Feature Map Consistency-enforcement (FMC). FDD reconstructs the shallow texture features of a thin student network to approximate the corresponding samples in a teacher network, which allows the high-resolution ones to directly guide the learning of the shallow features of the student network. FMC makes the size and direction of each deep feature map consistent between student and teacher networks, which constrains each pair of feature maps to produce the same feature distribution. FDD and FMC allow a thin student network to learn rich “privilege information” in feature maps of a wide teacher network. The overall performance of FMD is verified in multiple recognition tasks by comparing it with state-of-the-art knowledge distillation methods on low-resolution and noisy objects.



Z. Huang, S. Yang, M. C. Zhou, Z. Li, Z. Gong and Y. Chen, "Feature Map Distillation of Thin Nets for Low-resolution Object Recognition," in IEEE Transactions on Image Processing, doi: 10.1109/TIP.2022.3141255.
