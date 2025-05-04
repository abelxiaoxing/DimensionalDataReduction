import numpy as np
import glob
import os
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.special import wofz
from pandas import read_csv,to_numeric

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        raise NotImplementedError("Load method not implemented.")

class WineDataLoader(DataLoader):
    """红酒数据集加载器。"""
    def load(self):
        data = read_csv(self.file_path, header=None)
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:14].values
        return X, y

class MNISTDataLoader(DataLoader):
    """MNIST数据集加载器。"""
    def load(self):
        data = read_csv(self.file_path, delimiter='\t')
        labels = data.iloc[:, 0].str.split(':').str[1].astype(int).values
        features = data.iloc[:, 1:].values
        return features, labels

class RamanDataLoader(DataLoader):
    def __init__(self, data_dir, data_type='normal', raman_shift_range=(200, 1000), n_samples=1000):
        self.data_dir = data_dir
        self.data_type = data_type
        self.raman_shift_range = raman_shift_range
        self.n_samples = n_samples
        self.file_pattern = f'*__Raman_Data_{data_type}__*.txt'
        self.aligned_data = None
        self.labels = None
        self.label_names = None
        self.filtered_label_names = None
        self.standard_grid = np.linspace(raman_shift_range[0], raman_shift_range[1], n_samples)

    def load(self):
        print(f"正在加载{self.data_type}类型的拉曼光谱数据...")
        files = glob.glob(os.path.join(self.data_dir, self.file_pattern))
        print(f"找到{len(files)}个文件")
        aligned_spectra = []
        labels = []
        label_names = []
        for file_path in files:
            file_name = os.path.basename(file_path)
            material_name = file_name.split('__')[0]  # 提取文件名的第一部分作为物质名称
            if material_name not in label_names:
                label_names.append(material_name)
            label = label_names.index(material_name)
            spectrum = self._load_spectrum(file_path)
            aligned_spectrum = self._interpolate_spectrum(spectrum)
            if aligned_spectrum is not None:
                aligned_spectra.append(aligned_spectrum)
                labels.append(label)
        self.aligned_data = np.array(aligned_spectra)
        self.labels = np.array(labels)
        self.label_names = label_names
        
        print(f"数据读取完成，共有{len(self.aligned_data)}个样本，{len(self.label_names)}个类别")
        return self.aligned_data, self.labels
    
    def _load_spectrum(self, file_path):
        data = read_csv(file_path, skiprows=12, header=None, sep=',')
        data.columns = ['raman_shift', 'intensity']
        return data

    def voigt_profile(self, x, x0, alpha, gamma, A):
        return A * np.real(wofz((x - x0 + 1j*gamma)/(alpha*np.sqrt(2))))
    
    def _interpolate_spectrum(self, spectrum):
        spectrum = spectrum.apply(to_numeric, errors='coerce').dropna()
        raw_shift = spectrum['raman_shift'].values
        raw_intensity = spectrum['intensity'].values
        min_shift = float(np.min(raw_shift))
        max_shift = float(np.max(raw_shift))
        min_range = float(self.raman_shift_range[0])
        max_range = float(self.raman_shift_range[1])
        unique_shifts = np.unique(raw_shift)
        unique_intensities = np.array([raw_intensity[raw_shift == x].mean() for x in unique_shifts])
        needs_extrapolation = min_shift > min_range or max_shift < max_range
        
        if needs_extrapolation:
            # print(f"数据范围({min_shift:.2f}, {max_shift:.2f})不完全覆盖目标范围{self.raman_shift_range}，将采用Voigt轮廓外推。")
            cs = CubicSpline(unique_shifts, unique_intensities)
            aligned_intensity = np.zeros(self.n_samples)
            # 对于标准网格中在原始数据范围内的点，使用CubicSpline插值
            in_range_mask = (self.standard_grid >= min_shift) & (self.standard_grid <= max_shift)
            aligned_intensity[in_range_mask] = cs(self.standard_grid[in_range_mask])
            # 对于标准网格中超出原始数据范围的点，使用Voigt轮廓函数外推
            if min_shift > min_range:
                boundary_points = 10
                x_boundary = unique_shifts[:boundary_points]
                y_boundary = unique_intensities[:boundary_points]
                x0 = x_boundary[np.argmax(y_boundary)]
                A = np.max(y_boundary)
                alpha = 5.0
                gamma = 3.0
                low_range_mask = self.standard_grid < min_shift
                aligned_intensity[low_range_mask] = self.voigt_profile(self.standard_grid[low_range_mask], x0, alpha, gamma, A)
            
            if max_shift < max_range:
                boundary_points = 10
                x_boundary = unique_shifts[-boundary_points:]
                y_boundary = unique_intensities[-boundary_points:]
                x0 = x_boundary[np.argmax(y_boundary)]
                A = np.max(y_boundary)
                alpha = 5.0 
                gamma = 3.0
                high_range_mask = self.standard_grid > max_shift
                aligned_intensity[high_range_mask] = self.voigt_profile(self.standard_grid[high_range_mask], x0, alpha, gamma, A)
            
            return aligned_intensity

        # 使用CubicSpline进行对齐
        cs = CubicSpline(unique_shifts, unique_intensities)
        aligned_spectrum = cs(self.standard_grid)
        return aligned_spectrum
    
    def filter_classes(self, min_samples_per_class=2):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        valid_labels = unique_labels[counts >= min_samples_per_class]
        mask = np.isin(self.labels, valid_labels)
        filtered_data = self.aligned_data[mask]
        filtered_labels = self.labels[mask]
        label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_labels))}
        remapped_labels = np.array([label_map[label] for label in filtered_labels])
        filtered_label_names = [self.label_names[i] for i in valid_labels]
        self.filtered_label_names = filtered_label_names
        
        return filtered_data, remapped_labels, filtered_label_names