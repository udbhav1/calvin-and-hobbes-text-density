# Calvin and Hobbes Text Density

<p align="center">
    <img src="assets/strip.png" alt="Strip" style="width:100%;"/>
</p>

What's the distribution of (text_area/panel_area) for Calvin and Hobbes strips?

## Method

*The Complete Calvin and Hobbes* is available for free [here](https://ia600903.us.archive.org/20/items/TheCompleteCalvinHobbes_201902/The%20Complete%20Calvin%20%26%20Hobbes_text.pdf). With edge detection to find panel borders, structural observations to segment panels into strips, and OCR to extract text bounding boxes, we can annotate each page fairly accurately:

<p align="center">
    <img src="assets/annotated-page.png" alt="Annotated Page" style="width:100%;"/>
</p>

## Results

All panels, not segregated by strip:
![Ratio Distribution for Panels](assets/Text_Panel_Area_Ratio.png)
![Ratio Distribution for Panels without Outliers](assets/Text_Panel_Area_Ratio_without_Outliers.png)

Strips:
![Ratio Distribution for Strips](assets/Text_Strip_Area_Ratio.png)
![Ratio Distribution for Strips without Outliers](assets/Text_Strip_Area_Ratio_without_Outliers.png)

Normal distributions rule everything around me.
