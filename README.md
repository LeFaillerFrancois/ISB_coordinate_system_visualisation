# ISB coordinate system visualisation
3D python plots to visualise International Society of Biomechanics coordinate system of a marker kinematic file (.mat file). Actually a punch example. 
## Showcase
*The returned animation of the model with coordinate system*
<br>
![kinematic](results/kinematic_gif.gif)
<br>
*The mean punch trajectory animation*
<br>
![punch](results/mean_punch_gif.gif)
<br>
*The plot with only bones and markers (no grf, no coordinate system)*
<br>
![markers_and_bones](results/markers_and_bones.png)
*The plot with everything*
<br>
![everything](results/everything.png)
*The plots of all the punchs thrown and the mean punch trajectory*
<br>
![each_punch_and_mean_punch](results/each_punch_and_mean_punch.png)

## International Society of Biomechanics 
Tried to follow the recommandations for [upper body](https://pubmed.ncbi.nlm.nih.gov/15844264/) (Wu et al. 2005) and [lower body](https://pubmed.ncbi.nlm.nih.gov/11934426/) (Wu et al. 2002) from [ISB](https://isbweb.org/). The marker set used in our lab doesn't allow for exact following of the recommandations.

## Limitations 
Yet the COP GRF is not accurate. Euler angles calculation at the end were not compared with the ones of a known model/software such as Opensim.
<br>
Also we used many devices and sensors at the same time, that's why there are specific function in the python script to get the trigger and synch all the files (even if all of them aren't used in this script). 

## References
- Wu, G., Siegler, S., Allard, P., Kirtley, C., Leardini, A., Rosenbaum, D., Whittle, M., D'Lima, D. D., Cristofolini, L., Witte, H., Schmid, O., Stokes, I., & Standardization and Terminology Committee of the International Society of Biomechanics (2002). ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion--part I: ankle, hip, and spine. International Society of Biomechanics. Journal of biomechanics, 35(4), 543–548. https://doi.org/10.1016/s0021-9290(01)00222-6
- Wu, G., van der Helm, F. C., Veeger, H. E., Makhsous, M., Van Roy, P., Anglin, C., Nagels, J., Karduna, A. R., McQuade, K., Wang, X., Werner, F. W., Buchholz, B., & International Society of Biomechanics (2005). ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion--Part II: shoulder, elbow, wrist and hand. Journal of biomechanics, 38(5), 981–992. https://doi.org/10.1016/j.jbiomech.2004.05.042

