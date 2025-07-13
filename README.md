# Certified Robustness against Sensor Heterogeneity in Acoustic Sensing

## Abstract
Domain shifts due to microphone hardware heterogeneity pose challenges to machine learning-based acoustic sensing. Existing methods enhance empirical performance but lack theoretical understanding. This paper proposes Certified Adaptive Physics-informed transform (CertiAPT), an approach that provides formal certification on the model accuracy and improves empirical performance against microphone-induced domain shifts. CertiAPT incorporates a novel Adaptive Physics-informed Transform (APT) to create transformations toward the target microphone without requiring application samples collected by the target microphone. It also establishes a theoretical upper bound on accuracy degradation due to microphone characteristic differences on unseen microphones. Furthermore, a robust training method with an APT gradient update scheme leverages APT and certification constraints to tighten the upper bound and improve empirical accuracy across sensor conditions. Extensive experiments on three acoustic sensing tasks, including keyword spotting, room recognition, and automated speech recognition, validate CertiAPT’s certified robustness and show accuracy gains, compared with the latest approaches. 

## Citation

Coming soon ...

## Acknowledgments

[PhyAug](https://github.com/jiegev5/PhyAug)

