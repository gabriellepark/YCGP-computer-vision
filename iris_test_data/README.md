# Synthetic iris test data

5 grayscale eye images (280 rows x 320 cols, PNG) with known circle parameters.

- `images/iris_01.png` ... `iris_05.png`
- `localization.csv`: one row per image, columns image, pupil_x, pupil_y, pupil_r, iris_x, iris_y, iris_r
- `localization.json`: same data as pupil_info = [center_x, center_y, radius] and iris_info = [center_x, center_y, radius]

Coordinates: center_x = column, center_y = row, origin at top-left, y increases downward.
Pupil and iris centers are deliberately not identical.
Images are synthetic (not real eyes): useful for checking normalization logic, not model accuracy.
