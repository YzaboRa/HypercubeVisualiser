import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QLabel, QSizePolicy, QComboBox, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QFile
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from enum import Enum
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore",category=UserWarning)

dir_path = os.path.dirname(os.path.realpath(__file__))


WavelengthsPlacement = np.array([400,420,440,460,480,500,520,540, 560,580,600,620,640,660,680,700])


c1 = 'red'
c2 = 'cornflowerblue'

class Action(Enum):
    NONE = 0
    DRAG = 1
    RESIZE_TOP_LEFT = 2
    RESIZE_TOP_RIGHT = 3
    RESIZE_BOTTOM_LEFT = 4
    RESIZE_BOTTOM_RIGHT = 5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hypercube Visualizer")
        self.setGeometry(100, 100, 800, 700)  # Adjusted window width
        
        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a layout for the central widget
        layout = QVBoxLayout(central_widget)
        
        # Figures layout
        figures_layout = QHBoxLayout()
        
        # Left side: RGB image and Load button
        self.rgb_canvas = FigureCanvas(Figure(figsize=(6, 6)))  # Adjusted figure size
        figures_layout.addWidget(self.rgb_canvas)
        
        # Right side: Spectrum plot
        self.spectrum_canvas = FigureCanvas(Figure(figsize=(6, 6)))  # Adjusted figure size
        figures_layout.addWidget(self.spectrum_canvas)
        
        layout.addLayout(figures_layout)
        
        # Add Navigation Toolbars
        toolbar_layout = QHBoxLayout()
        
        self.rgb_toolbar = NavigationToolbar(self.rgb_canvas, self)
        toolbar_layout.addWidget(self.rgb_toolbar)
        
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        toolbar_layout.addWidget(self.spectrum_toolbar)
        
        layout.addLayout(toolbar_layout)
        
        # Load the UI elements from the UI file
        ui_layout = QHBoxLayout()
        
        left_side_layout = QVBoxLayout()
        self.load_hypercube_button = QPushButton("Load Hypercube")
        left_side_layout.addWidget(self.load_hypercube_button)
        self.load_hypercube_button.clicked.connect(self.load_hypercube)
        
        self.load_wavelengths_button = QPushButton("Load Wavelengths")
        left_side_layout.addWidget(self.load_wavelengths_button)
        self.load_wavelengths_button.clicked.connect(self.load_wavelengths)
        
        ui_layout.addLayout(left_side_layout)
        
        middle_layout = QVBoxLayout()
        self.display_rgb_button = QPushButton("Display RGB")
        middle_layout.addWidget(self.display_rgb_button)
        self.display_rgb_button.clicked.connect(self.update_rgbplot)
        
        rgb_layout = QHBoxLayout()
        self.red_wav_combo = QComboBox()
        rgb_layout.addWidget(QLabel("Red:"))
        rgb_layout.addWidget(self.red_wav_combo)
        self.red_wav_combo.currentIndexChanged.connect(self.update_rgbplot)
        
        self.green_wav_combo = QComboBox()
        rgb_layout.addWidget(QLabel("Green:"))
        rgb_layout.addWidget(self.green_wav_combo)
        self.green_wav_combo.currentIndexChanged.connect(self.update_rgbplot)
        
        self.blue_wav_combo = QComboBox()
        rgb_layout.addWidget(QLabel("Blue:"))
        rgb_layout.addWidget(self.blue_wav_combo)
        self.blue_wav_combo.currentIndexChanged.connect(self.update_rgbplot)
        
        middle_layout.addLayout(rgb_layout)

        ui_layout.addLayout(middle_layout)
        
        right_side_layout = QVBoxLayout()
        self.display_wavelength_button = QPushButton("Display Wavelength")
        right_side_layout.addWidget(self.display_wavelength_button)
        self.display_wavelength_button.clicked.connect(self.display_wavelength_image)
        
        self.selected_wavelength_combo = QComboBox()
        right_side_layout.addWidget(QLabel("Select Wavelength:"))
        right_side_layout.addWidget(self.selected_wavelength_combo)
        ui_layout.addLayout(right_side_layout)
        # self.selected_wavelength_combo.currentIndexChanged.connect(self.display_wavelength_image)
        
        rescale_image_layout = QVBoxLayout()
        self.rescale_image_button = QPushButton("Rescale Image")
        rescale_image_layout.addWidget(self.rescale_image_button)
        self.rescale_image_button.clicked.connect(self.rescale_image_button_clicked)
        
        self.rescale_image_box = QLineEdit()
        rescale_image_layout.addWidget(QLabel("Factor:"))
        rescale_image_layout.addWidget(self.rescale_image_box)

        ui_layout.addLayout(rescale_image_layout)

        
        rescale_spectra_layout = QVBoxLayout()
        self.rescale_spectra_button = QPushButton("Rescale Spectra")
        rescale_spectra_layout.addWidget(self.rescale_spectra_button)
        self.rescale_spectra_button.clicked.connect(self.rescale_spectra_button_clicked)
        
        self.rescale_spectra_box = QLineEdit()
        rescale_spectra_layout.addWidget(QLabel("Factor:"))
        rescale_spectra_layout.addWidget(self.rescale_spectra_box)
        ui_layout.addLayout(rescale_spectra_layout)
        
        refpatch_layout = QVBoxLayout()
        self.reference_patch_combo = QComboBox()
        self.reference_patch_combo2 = QComboBox()
        refpatch_layout.addWidget(QLabel("Selected Reference Patch:"))
        refpatch_layout.addWidget(self.reference_patch_combo)
        refpatch_layout.addWidget(self.reference_patch_combo2)
        self.reference_patch_combo.currentIndexChanged.connect(self.update_spectraplot)
        self.reference_patch_combo2.currentIndexChanged.connect(self.update_spectraplot)

        ui_layout.addLayout(refpatch_layout)
        
        layout.addLayout(ui_layout)


        # Create ROI rectangles
        self.roi_rect1 = Rectangle((0, 0), 50, 50, edgecolor='red', facecolor='none')
        self.roi_rect2 = Rectangle((10, 10), 50, 50, edgecolor='cornflowerblue', facecolor='none')
        self.rgb_canvas.figure.gca().add_patch(self.roi_rect1)
        self.rgb_canvas.figure.gca().add_patch(self.roi_rect2)
        self.roi_rect1.set_visible(True)
        self.roi_rect2.set_visible(True)

        # Connect mouse events for ROI interaction
        self.rgb_canvas.mpl_connect('button_press_event', self.on_press)
        self.rgb_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.rgb_canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False
        self.start_x = None
        self.start_y = None
        self.current_roi = None

        # Add attributes to hold references to image plots
        self.rgb_image_plot = None
        self.wavelength_image_plot = None

        self.ax_spectrum = self.spectrum_canvas.figure.add_subplot(111)
        self.ax_rgb = self.rgb_canvas.figure.add_subplot(111)

        self.action = None

        # Initialize variables
        self.hypercube = None
        self.wavelengths = None
        self.reference_spectra = None
        self.IsThereCB = False
        self.rescale_value = 1
        self.rescale_spectra = 1


    def load_hypercube(self):
        # Prompt user to select the hypercube file
        hypercube_path, _ = QFileDialog.getOpenFileName(self, "Select Hypercube File", "", "NPZ files (*.npz)")
        # # Load hypercube data
        self.reference_spectra_path = dir_path+'/Micro_Nano_CheckerTargetData.xls'
        self.reference_rgb_path = dir_path+'/Macbeth_Adobe.xlsx'

        if hypercube_path:
            hypercube = np.load(hypercube_path)['arr_0']
            self.hypercube = hypercube
            if np.nanmax(self.hypercube)>1:
                print(f'\nHypercube is not normalised: {np.amax(self.hypercube)} for \n  {hypercube_path}')
                print(f'Normalising it...')
                self.hypercube = self.hypercube/np.nanmax(self.hypercube)
            if self.wavelengths is None:
                # wavelengths = np.load(wavelengths_path)['arr_0']
                wavelengths = WavelengthsPlacement
                self.wavelengths = wavelengths

            # Load reference spectra
            reference_spectra = np.array(pd.read_excel(self.reference_spectra_path, sheet_name='Spectra')) 
            reference_rgb = np.array(pd.read_excel(self.reference_rgb_path))
            idx_min = self.find_closest(reference_spectra[:,0], self.wavelengths[0])
            idx_max = self.find_closest(reference_spectra[:,0], self.wavelengths[-1])
            self.reference_spectra = reference_spectra[idx_min:idx_max,:]
            self.reference_rgb = reference_rgb
            self.Nwhite = 8
            # print(self.reference_spectra)

            # Populate reference patch combo box
            self.reference_patch_combo.clear()
            ItemsList = [str(i) for i in range(1, reference_spectra.shape[1])]
            self.reference_patch_combo.addItems(ItemsList)

            self.reference_patch_combo2.clear()
            self.reference_patch_combo2.addItems(ItemsList)

            # Populate wavelength combo box
            self.selected_wavelength_combo.clear()
            self.selected_wavelength_combo.addItems([str(wavelength) for wavelength in self.wavelengths])

            # Populate RGB selection combo boxes
            red_pos = self.find_closest(self.wavelengths,630)
            green_pos = self.find_closest(self.wavelengths,530)
            blue_pos = self.find_closest(self.wavelengths,470)

            self.red_wav_combo.clear()
            self.red_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.red_wav_combo.setCurrentIndex(red_pos)

            self.green_wav_combo.clear()
            self.green_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.green_wav_combo.setCurrentIndex(green_pos)

            self.blue_wav_combo.clear()
            self.blue_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.blue_wav_combo.setCurrentIndex(blue_pos)

            self.update_rgbplot()

    def load_wavelengths(self):
        # Prompt user to select the wavelengths file
        wavelengths_path, _ = QFileDialog.getOpenFileName(self, "Select Wavelengths File", "", "NPZ files (*.npz)")
        if wavelengths_path:
            wavelengths = np.load(wavelengths_path)['arr_0']
            self.wavelengths = wavelengths
            # Populate wavelength combo box
            self.selected_wavelength_combo.clear()
            self.selected_wavelength_combo.addItems([str(wavelength) for wavelength in self.wavelengths])


            # Populate RGB selection combo boxes
            red_pos = self.find_closest(self.wavelengths,630)
            green_pos = self.find_closest(self.wavelengths,530)
            blue_pos = self.find_closest(self.wavelengths,470)

            self.red_wav_combo.clear()
            self.red_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.red_wav_combo.setCurrentIndex(red_pos)

            self.green_wav_combo.clear()
            self.green_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.green_wav_combo.setCurrentIndex(green_pos)

            self.blue_wav_combo.clear()
            self.blue_wav_combo.addItems([str(wavelength) for wavelength in self.wavelengths])
            self.blue_wav_combo.setCurrentIndex(blue_pos)

            # Load reference spectra
            reference_spectra = np.array(pd.read_excel(self.reference_spectra_path, sheet_name='Spectra')) 
            reference_rgb = np.array(pd.read_excel(self.reference_rgb_path))
            idx_min = self.find_closest(reference_spectra[:,0], self.wavelengths[0])
            idx_max = self.find_closest(reference_spectra[:,0], self.wavelengths[-1])
            self.reference_spectra = reference_spectra[idx_min:idx_max,:]
            self.reference_rgb = reference_rgb
            self.Nwhite = 8

            self.update_rgbplot()
            self.update_spectraplot()

    
    def construct_rgb_image(self):
        # Code to reconstruct RGB image from hypercube
        hypercube = self.hypercube
        NN, YY, XX = hypercube.shape

        ## Check which wavelengths the users wants to use for each channel
        red_pos =self.red_wav_combo.currentIndex()
        green_pos =self.green_wav_combo.currentIndex()
        blue_pos =self.blue_wav_combo.currentIndex()

        im_red = hypercube[red_pos, :,:]
        im_green = hypercube[green_pos, :,:]
        im_blue = hypercube[blue_pos, :,:]
        imRGB = np.zeros((YY,XX,3))
        for i in range(0,XX):
            for j in range(0,YY):
                imRGB[j,i,0] = im_red[j,i]
                imRGB[j,i,1] = im_green[j,i]
                imRGB[j,i,2] = im_blue[j,i]

        rgb_image = imRGB
        return rgb_image

    def update_rgbplot(self):
        if self.hypercube is None or self.wavelengths is None:
            # print("Hypercube or wavelengths data not loaded.")
            return

        try:
            rgb_image = self.construct_rgb_image()
            self.wavelength_image_plot = None
            if self.rescale_value != 1:
                rgb_image = self.BrightenImage(rgb_image, self.rescale_value)

            if self.rgb_image_plot is None:
                self.rgb_image_plot = self.ax_rgb.imshow(rgb_image)
            else:
                self.rgb_image_plot.set_data(rgb_image)

            self.ax_rgb.set_axis_off()
            if self.rescale_value == 1:
                self.ax_rgb.set_title('Simulated RGB image')
            else:
                self.ax_rgb.set_title(f'Simulated RGB image, rescaled {self.rescale_value}')
            self.rgb_canvas.figure.tight_layout()

            self.rgb_canvas.draw()

            self.roi_rect1.set_visible(True)
            self.roi_rect2.set_visible(True)

        except Exception as e:
            print(f"Failed to update RGB plot: {e}")


    def calculate_roi_spectrum(self, roi_rect):
        # Extract ROI coordinates
        x, y = roi_rect.get_x(), roi_rect.get_y()
        width, height = roi_rect.get_width(), roi_rect.get_height()

        x, y = int(np.round(x, 0)), int(np.round(y, 0))
        width, height = int(np.round(width, 0)), int(np.round(height, 0))
        spectrum_roi = self.hypercube[:, y:y + height, x:x + width]

        # Calculate the average spectrum
        average_spectrum = np.mean(spectrum_roi, axis=(1, 2))
        return average_spectrum


    def display_wavelength_image(self):
        # Display image corresponding to selected wavelength
        selected_wavelength = float(self.selected_wavelength_combo.currentText())
        idx = np.argmin(np.abs(self.wavelengths - selected_wavelength))
        wavelength_image = self.hypercube[idx, :, :]

        if self.wavelength_image_plot is None:
            self.wavelength_image_plot = self.ax_rgb.imshow(wavelength_image, cmap='gray')
            self.rgb_image_plot = None
        else:
            self.wavelength_image_plot.set_data(wavelength_image)
            self.rgb_image_plot = None

        self.ax_rgb.set_axis_off()
        self.ax_rgb.set_title(f'Image at Wavelength {selected_wavelength}')
        self.rgb_canvas.figure.tight_layout()
        self.rgb_canvas.draw()


        
    def update_spectraplot(self):
        if self.hypercube is None or self.wavelengths is None:
            # print("Hypercube or wavelengths data not loaded.")
            return

        try: 
            self.ax_spectrum.clear()
            # Plot selected reference patch spectrum
            if self.reference_spectra is not None:
                selected_patch_index = int(self.reference_patch_combo.currentIndex()) + 1
                selected_patch_spectrum = self.reference_spectra[:, selected_patch_index]
                selected_patch_spectrumN = np.divide(selected_patch_spectrum, self.reference_spectra[:, self.Nwhite])

                selected_patch_index2 = int(self.reference_patch_combo2.currentIndex()) + 1
                selected_patch_spectrum2 = self.reference_spectra[:, selected_patch_index2]
                selected_patch_spectrum2N = np.divide(selected_patch_spectrum2, self.reference_spectra[:, self.Nwhite])

                selected_patch_wavelengths = self.reference_spectra[:, 0]
                cc = (self.reference_rgb[selected_patch_index-1, 0]/255, self.reference_rgb[selected_patch_index-1, 1]/255, self.reference_rgb[selected_patch_index-1, 2]/255)
                cc2 = (self.reference_rgb[selected_patch_index2-1, 0]/255, self.reference_rgb[selected_patch_index2-1, 1]/255, self.reference_rgb[selected_patch_index2-1, 2]/255)
                self.ax_spectrum.plot(selected_patch_wavelengths, selected_patch_spectrumN, ls='solid', lw=5, color=cc, alpha=1, label=f'Patch {selected_patch_index}')
                self.ax_spectrum.plot(selected_patch_wavelengths, selected_patch_spectrum2N, ls='solid', lw=5, color=cc2, alpha=1, label=f'Patch {selected_patch_index2}')

            # Plot ROI spectra
            if self.roi_rect1.get_visible():
                roi_spectrum1 = self.calculate_roi_spectrum(self.roi_rect1)
                self.ax_spectrum.plot(self.wavelengths, roi_spectrum1*self.rescale_spectra, '.-', color=c1)
            if self.roi_rect2.get_visible():
                roi_spectrum2 = self.calculate_roi_spectrum(self.roi_rect2)
                self.ax_spectrum.plot(self.wavelengths, roi_spectrum2*self.rescale_spectra, '.-', color=c2)

            self.ax_spectrum.set_xlabel("Wavelength")
            self.ax_spectrum.set_ylabel("Intensity")
            self.ax_spectrum.set_title("Spectrum in ROIs")
            self.ax_spectrum.legend()
            self.spectrum_canvas.figure.tight_layout()
            self.spectrum_canvas.draw()
        except Exception as e:
            print(f"Failed to update spectral plot: {e}")


    ## Function to find the closest point in the dataset to a given value
    def find_closest(self, arr, val):
        idx = np.abs(arr - val).argmin()
        return idx


    ## Handle rescale of the RGB image
    def rescale_image_button_clicked(self):
        if self.rescale_image_box.text():
            self.rescale_value = float(self.rescale_image_box.text())
            self.update_rgbplot()

    ## Handle rescale of the spectra
    def rescale_spectra_button_clicked(self):
        if self.rescale_spectra_box.text():
            self.rescale_spectra = float(self.rescale_spectra_box.text())
            self.update_spectraplot()

    ## Rescale the RGB image
    def BrightenImage(self, imRGB, Scale):
        im0 = imRGB*Scale
        pos = np.where(im0>1.0)
        for k in range(0,len(pos[0])):
            im0[pos[0][k], pos[1][k], pos[2][k]] = 1.0
        return im0

    ## Reszie figures as the GUI window is modified    
    def resizeEvent(self, event):
        new_size = event.size()
        self.rgb_canvas.setGeometry(0, 0, int(new_size.width()/2), new_size.height() - 100)  # Adjust the height as needed
        self.spectrum_canvas.setGeometry(int(new_size.width()/2), 0, int(new_size.width()/2), new_size.height() - 100)


    def on_press(self, event):
        x = event.xdata
        y = event.ydata
        tol = 5

        if self.roi_rect1.contains(event)[0]:
            self.dragging = True
            self.start_x = self.roi_rect1.get_x() - x
            self.start_y = self.roi_rect1.get_y() - y
            self.current_roi = self.roi_rect1
            self.action = Action.DRAG
        elif self.roi_rect2.contains(event)[0]:
            self.dragging = True
            self.start_x = self.roi_rect2.get_x() - x
            self.start_y = self.roi_rect2.get_y() - y
            self.current_roi = self.roi_rect2
            self.action = Action.DRAG
        else:
            self.dragging = False
            self.action = Action.NONE
            self.current_roi = None

        if self.current_roi:
            width = self.current_roi.get_width()
            height = self.current_roi.get_height()
            if x < self.current_roi.get_x() + tol and y < self.current_roi.get_y() + tol:
                self.action = Action.RESIZE_TOP_LEFT
            elif x > self.current_roi.get_x() + width - tol and y < self.current_roi.get_y() + tol:
                self.action = Action.RESIZE_TOP_RIGHT
            elif x < self.current_roi.get_x() + tol and y > self.current_roi.get_y() + height - tol:
                self.action = Action.RESIZE_BOTTOM_LEFT
            elif x > self.current_roi.get_x() + width - tol and y > self.current_roi.get_y() + height - tol:
                self.action = Action.RESIZE_BOTTOM_RIGHT

            if self.action != Action.DRAG:
                self.start_x = self.current_roi.get_x() + width
                self.start_y = self.current_roi.get_y() + height


    def on_motion(self, event):
        if self.dragging:
            if self.action == Action.DRAG:
                x = event.xdata
                y = event.ydata
                if x is not None and y is not None:
                    self.current_roi.set_x(x + self.start_x)
                    self.current_roi.set_y(y + self.start_y)
                    self.rgb_canvas.draw()
            elif self.action != Action.NONE:
                x = event.xdata
                y = event.ydata
                if x is not None and y is not None:
                    new_x = self.start_x
                    new_y = self.start_y
                    if self.action in (Action.RESIZE_TOP_LEFT, Action.RESIZE_BOTTOM_LEFT):
                        new_x = min(x, new_x)
                    if self.action in (Action.RESIZE_TOP_LEFT, Action.RESIZE_TOP_RIGHT):
                        new_y = min(y, new_y)
                    if self.action in (Action.RESIZE_TOP_RIGHT, Action.RESIZE_BOTTOM_RIGHT):
                        new_x = max(x, new_x)
                    if self.action in (Action.RESIZE_BOTTOM_LEFT, Action.RESIZE_BOTTOM_RIGHT):
                        new_y = max(y, new_y)
                    width = abs(new_x - self.start_x)
                    height = abs(new_y - self.start_y)
                    
                    # Check if the new width or height is too small
                    min_width = 10  # Minimum width threshold
                    min_height = 10  # Minimum height threshold
                    if width < min_width or height < min_height:
                        # Resize the ROI back to its previous size
                        self.current_roi.set_x(self.start_x - self.current_roi.get_width())
                        self.current_roi.set_y(self.start_y - self.current_roi.get_height())
                        self.current_roi.set_width(max(width, min_width))
                        self.current_roi.set_height(max(height, min_height))
                    else:
                        # Update the ROI size
                        self.current_roi.set_x(min(new_x, self.start_x))
                        self.current_roi.set_y(min(new_y, self.start_y))
                        self.current_roi.set_width(width)
                        self.current_roi.set_height(height)

                    self.rgb_canvas.draw()
        else:
            self.dragging = False
            self.action = Action.NONE
            self.update_spectraplot()




    def on_release(self, event):
        self.dragging = False
        self.action = Action.NONE
        self.update_spectraplot()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())




