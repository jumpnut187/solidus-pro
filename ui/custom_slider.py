from ui.qt import (
    QColor,
    QPainter,
    QPen,
    QRect,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    Qt,
)

class MarkedSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.marker_position = 30
        self.marker_color = QColor(255, 0, 0)
        self.marker_width = 2

    def setMarkerPosition(self, position):
        self.marker_position = position
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        val_range = self.maximum() - self.minimum()
        if val_range == 0:
            return

        marker_val = self.minimum() + (self.marker_position / 100.0) * val_range

        if self.orientation() == Qt.Horizontal:
            slider_length = groove_rect.width()
            slider_min_pos = groove_rect.left()

            x_pos = slider_min_pos + int((self.marker_position / 100.0) * slider_length)

            pen = QPen(self.marker_color)
            pen.setWidth(self.marker_width)
            painter.setPen(pen)

            top = groove_rect.top() - 2
            bottom = groove_rect.bottom() + 2
            painter.drawLine(x_pos, top, x_pos, bottom)

        else:
            slider_length = groove_rect.height()
            slider_min_pos = groove_rect.bottom()

            y_pos = slider_min_pos - int((self.marker_position / 100.0) * slider_length)

            pen = QPen(self.marker_color)
            pen.setWidth(self.marker_width)
            painter.setPen(pen)

            left = groove_rect.left() - 2
            right = groove_rect.right() + 2
            painter.drawLine(left, y_pos, right, y_pos)
