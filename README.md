Использованные работы (помимо учебников):
1. Matsushima, K., & Shimobaba, T. (2009). «Band-limited angular spectrum method for numerical simulation of free-space propagation in the near field.» Optics Express, 17(22), 19662–19673.
2. Дифракция на произвольных фазовых масках - вместо бинарных апертур (либо материя либо пространство), произвольная фазовая функция - возможность интегрировать спирали, дифракционные линзы и тд. (Vellekoop, I. M., & Mosk, A. P. (2007). «Focusing coherent light through opaque strongly scattering media.» Optics Letters, 32(16), 2309–2311.)

Дальнейшее развитие:
1. Zero-padding для интерполяции - увеличение расширения выходной картинки без увеличиние входной сетки. (Mehrabkhani, P., & Schneider, T. (2017). «Is there a reason to use zero-padding in diffraction pattern calculations?» Optics Express, 25(22), 27290–27303.)
набросок реализации: массив размером `N_padded=N*pad_factor (being 4 or 8 or smth else)` -> Скопировать апертуру в центр, остальное нули -> fft from padded -> выходной шаг сетки уменьшается в `padfactor`
3. Частично когерентный свет (РЕАЛИЗМ) - реализация реального света, частично когерентного ( Goodman, J. W. (2015). «Statistical Optics» (2nd ed.). Wiley. Глава 7.)
набросок реализации: добавить некоторую погрешность к волнам, результат усреднять
# Diffraction_web
