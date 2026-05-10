from PyQt5.QtWidgets import QSlider, QStyle, QStyleOptionSlider
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QPen

class MarkedSlider(QSlider):
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.marker_position = 30  # Default marker at 30%
        self.marker_color = QColor(255, 0, 0)  # Red color
        self.marker_width = 2

    def setMarkerPosition(self, position):
        """Set the position of the marker (0-100)"""
        self.marker_position = position
        self.update()

    def paintEvent(self, event):
        """Override paint event to draw the marker line"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        
        # Calculate pixel position for the marker
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        # Get the slider groove rect (the track)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        
        # Calculate position based on range and marker percentage
        # Correctly handle min/max of the slider
        val_range = self.maximum() - self.minimum()
        if val_range == 0:
            return
            
        # 30% relative to the range, added to minimum
        marker_val = self.minimum() + (self.marker_position / 100.0) * val_range
        
        # Convert value to pixel position
        # Using built-in style method to get handle position for a specific value would be ideal,
        # but manual calculation is often necessary for custom overlays on the groove.
        
        # Simple interpolation logic for horizontal slider
        if self.orientation() == Qt.Horizontal:
            slider_length = groove_rect.width()
            slider_min_pos = groove_rect.left()
            
            # Account for handle width to be precise (scrol_pos is usually center of handle)
            # But for a simple groove marker, strict linear interpolation over groove width is usually close enough visually
            # For higher precision, we should consider the handle size.
            
            # Let's map 0-100% to the groove rect
            x_pos = slider_min_pos + int((self.marker_position / 100.0) * slider_length)
            
            # Draw line
            pen = QPen(self.marker_color)
            pen.setWidth(self.marker_width)
            painter.setPen(pen)
            
            # Draw vertical line across the slider height (or just the groove height)
            # Drawing slightly larger than groove to be visible
            top = groove_rect.top() - 2
            bottom = groove_rect.bottom() + 2
            painter.drawLine(x_pos, top, x_pos, bottom)
            
        else:
            # Vertical slider support if needed
            slider_length = groove_rect.height()
            slider_min_pos = groove_rect.bottom() # Vertical sliders usually go bottom-up
            
            # PyQt vertical sliders: min is at bottom
            y_pos = slider_min_pos - int((self.marker_position / 100.0) * slider_length)
            
            pen = QPen(self.marker_color)
            pen.setWidth(self.marker_width)
            painter.setPen(pen)
            
            left = groove_rect.left() - 2
            right = groove_rect.right() + 2
            painter.drawLine(left, y_pos, right, y_pos)
