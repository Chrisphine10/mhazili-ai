"""
Vision Service
Uses YOLO/DETR + OpenCV to understand screen and element locations
Provides computer vision capabilities for GUI automation
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import base64
from pathlib import Path
import json

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image, ImageDraw, ImageFont
    import pyautogui
except ImportError as e:
    logging.warning(f"Vision service dependencies not available: {e}")
    torch = None
    transforms = None
    Image = None
    pyautogui = None


class DetectionModel(Enum):
    YOLO = "yolo"
    DETR = "detr"
    TEMPLATE = "template"
    OCR = "ocr"


class ElementType(Enum):
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    LINK = "link"
    IMAGE = "image"
    ICON = "icon"
    WINDOW = "window"
    MENU = "menu"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.x + self.width, self.y + self.height)
    
    def overlaps_with(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        """Check if this bbox overlaps with another by given threshold"""
        x_overlap = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y_overlap = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        
        overlap_area = x_overlap * y_overlap
        self_area = self.width * self.height
        other_area = other.width * other.height
        
        return overlap_area / min(self_area, other_area) >= threshold


@dataclass
class DetectedElement:
    """Detected GUI element"""
    element_type: ElementType
    confidence: float
    bounding_box: BoundingBox
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None
    template_name: Optional[str] = None


@dataclass
class ScreenAnalysis:
    """Complete screen analysis result"""
    screenshot_path: str
    elements: List[DetectedElement]
    timestamp: float
    screen_size: Tuple[int, int]
    analysis_duration: float


class VisionService:
    """
    Computer vision service for GUI automation
    Supports multiple detection models and methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model settings
        self.default_model = DetectionModel(self.config.get('default_model', 'template'))
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.device = self._get_device()
        
        # Paths
        self.models_path = Path(self.config.get('models_path', 'models'))
        self.screenshots_path = Path(self.config.get('screenshots_path', 'screenshots'))
        self.templates_path = Path(self.config.get('templates_path', 'templates'))
        
        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.screenshots_path.mkdir(exist_ok=True)
        self.templates_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self._load_models()
        
        # OCR settings
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        self.ocr_reader = None
        self._init_ocr()
        
        # Template element type mapping
        self.template_type_mapping = self._load_template_mapping()
    
    def _get_device(self) -> str:
        """Determine the best device to use for inference"""
        if torch and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def _load_models(self):
        """Load detection models"""
        try:
            # Load YOLO model
            if self.config.get('yolo_enabled', True):
                self._load_yolo_model()
            
            # Load DETR model  
            if self.config.get('detr_enabled', False):
                self._load_detr_model()
                
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
    
    def _load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            # This would load a custom-trained YOLO model for GUI elements
            # For now, we'll use a placeholder
            self.logger.info("YOLO model loading placeholder - would load custom GUI model")
            self.models[DetectionModel.YOLO] = None
        except Exception as e:
            self.logger.error(f"YOLO model loading failed: {e}")
    
    def _load_detr_model(self):
        """Load DETR model for object detection"""
        try:
            # This would load a DETR model fine-tuned for GUI elements
            self.logger.info("DETR model loading placeholder - would load custom GUI model")
            self.models[DetectionModel.DETR] = None
        except Exception as e:
            self.logger.error(f"DETR model loading failed: {e}")
    
    def _init_ocr(self):
        """Initialize OCR capabilities"""
        try:
            if self.ocr_enabled:
                # Try to import and initialize OCR libraries
                try:
                    import easyocr
                    self.ocr_reader = easyocr.Reader(['en'])
                    self.logger.info("EasyOCR initialized")
                except ImportError:
                    try:
                        import pytesseract
                        self.ocr_reader = pytesseract
                        self.logger.info("Tesseract OCR initialized")
                    except ImportError:
                        self.logger.warning("No OCR library available")
                        self.ocr_reader = None
        except Exception as e:
            self.logger.error(f"OCR initialization failed: {e}")
            self.ocr_reader = None
    
    def _load_template_mapping(self) -> Dict[str, ElementType]:
        """Load template to element type mapping"""
        mapping_file = self.templates_path / "element_mapping.json"
        
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    return {k: ElementType(v) for k, v in mapping_data.items()}
            except Exception as e:
                self.logger.error(f"Failed to load template mapping: {e}")
        
        # Default mapping based on filename patterns
        return {
            "button": ElementType.BUTTON,
            "btn": ElementType.BUTTON,
            "text_field": ElementType.TEXT_FIELD,
            "input": ElementType.TEXT_FIELD,
            "dropdown": ElementType.DROPDOWN,
            "select": ElementType.DROPDOWN,
            "checkbox": ElementType.CHECKBOX,
            "check": ElementType.CHECKBOX,
            "radio": ElementType.RADIO_BUTTON,
            "link": ElementType.LINK,
            "icon": ElementType.ICON,
            "menu": ElementType.MENU,
        }
    
    # =================== SCREENSHOT OPERATIONS ===================
    
    def take_screenshot(self, save_path: Optional[str] = None) -> str:
        """
        Take a screenshot of the current screen
        
        Args:
            save_path: Optional path to save screenshot
            
        Returns:
            str: Path to saved screenshot
        """
        try:
            if not pyautogui:
                raise ImportError("PyAutoGUI not available")
                
            # Generate filename if not provided
            if save_path is None:
                timestamp = int(time.time())
                save_path = str(self.screenshots_path / f"screenshot_{timestamp}.png")
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(save_path)
            
            self.logger.info(f"Screenshot saved: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            raise
    
    def take_region_screenshot(self, bbox: BoundingBox, save_path: Optional[str] = None) -> str:
        """
        Take screenshot of a specific region
        
        Args:
            bbox: Bounding box of region to capture
            save_path: Optional path to save screenshot
            
        Returns:
            str: Path to saved screenshot
        """
        try:
            if not pyautogui:
                raise ImportError("PyAutoGUI not available")
                
            if save_path is None:
                timestamp = int(time.time())
                save_path = str(self.screenshots_path / f"region_{timestamp}.png")
            
            # Take region screenshot
            screenshot = pyautogui.screenshot(region=(bbox.x, bbox.y, bbox.width, bbox.height))
            screenshot.save(save_path)
            
            self.logger.info(f"Region screenshot saved: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Region screenshot failed: {e}")
            raise
    
    # =================== ELEMENT DETECTION ===================
    
    def analyze_screen(self, model: DetectionModel = None) -> ScreenAnalysis:
        """
        Perform complete screen analysis
        
        Args:
            model: Detection model to use
            
        Returns:
            ScreenAnalysis: Complete analysis results
        """
        start_time = time.time()
        
        try:
            # Take screenshot
            screenshot_path = self.take_screenshot()
            
            # Get screen size
            if pyautogui:
                screen_size = pyautogui.size()
            else:
                screen_size = (1920, 1080)
            
            # Detect elements
            model = model or self.default_model
            elements = self.detect_elements(screenshot_path, model)
            
            # Create analysis result
            analysis = ScreenAnalysis(
                screenshot_path=screenshot_path,
                elements=elements,
                timestamp=time.time(),
                screen_size=screen_size,
                analysis_duration=time.time() - start_time
            )
            
            self.logger.info(f"Screen analysis completed in {analysis.analysis_duration:.2f}s, found {len(elements)} elements")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Screen analysis failed: {e}")
            raise
    
    def detect_elements(self, image_path: str, model: DetectionModel = None) -> List[DetectedElement]:
        """
        Detect GUI elements in an image
        
        Args:
            image_path: Path to image file
            model: Detection model to use
            
        Returns:
            List[DetectedElement]: Detected elements
        """
        model = model or self.default_model
        
        try:
            if model == DetectionModel.YOLO:
                return self._detect_with_yolo(image_path)
            elif model == DetectionModel.DETR:
                return self._detect_with_detr(image_path)
            elif model == DetectionModel.TEMPLATE:
                return self._detect_with_template_matching(image_path)
            elif model == DetectionModel.OCR:
                return self._detect_with_ocr(image_path)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            self.logger.error(f"Element detection failed: {e}")
            return []
    
    def _detect_with_yolo(self, image_path: str) -> List[DetectedElement]:
        """Detect elements using YOLO model"""
        # Placeholder for YOLO detection
        # Would use a custom-trained YOLO model for GUI elements
        self.logger.info("YOLO detection placeholder")
        return []
    
    def _detect_with_detr(self, image_path: str) -> List[DetectedElement]:
        """Detect elements using DETR model"""
        # Placeholder for DETR detection
        # Would use a fine-tuned DETR model for GUI elements
        self.logger.info("DETR detection placeholder")
        return []
    
    def _detect_with_template_matching(self, image_path: str) -> List[DetectedElement]:
        """Detect elements using template matching"""
        elements = []
        
        try:
            # Load main image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Search for all templates in templates directory
            for template_path in self.templates_path.glob("*.png"):
                template_elements = self._match_template(image, str(template_path))
                elements.extend(template_elements)
            
            # Remove overlapping detections
            elements = self._remove_overlapping_elements(elements)
            
            self.logger.info(f"Template matching found {len(elements)} elements")
            return elements
            
        except Exception as e:
            self.logger.error(f"Template matching failed: {e}")
            return []
    
    def _match_template(self, image: np.ndarray, template_path: str) -> List[DetectedElement]:
        """Match a single template against the image"""
        elements = []
        
        try:
            # Load template
            template = cv2.imread(template_path)
            if template is None:
                return elements
            
            template_name = Path(template_path).stem
            
            # Perform template matching
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where matching exceeds threshold
            locations = np.where(result >= self.confidence_threshold)
            
            for pt in zip(*locations[::-1]):  # Switch x and y
                # Create bounding box
                bbox = BoundingBox(
                    x=pt[0],
                    y=pt[1],
                    width=template.shape[1],
                    height=template.shape[0]
                )
                
                # Get confidence score
                confidence = result[pt[1], pt[0]]
                
                # Determine element type from template name
                element_type = self._get_element_type_from_template(template_name)
                
                # Create detected element
                element = DetectedElement(
                    element_type=element_type,
                    confidence=float(confidence),
                    bounding_box=bbox,
                    template_name=template_name
                )
                
                elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Template matching failed for {template_path}: {e}")
            return []
    
    def _get_element_type_from_template(self, template_name: str) -> ElementType:
        """Determine element type from template name"""
        template_name_lower = template_name.lower()
        
        for keyword, element_type in self.template_type_mapping.items():
            if keyword in template_name_lower:
                return element_type
        
        return ElementType.UNKNOWN
    
    def _remove_overlapping_elements(self, elements: List[DetectedElement]) -> List[DetectedElement]:
        """Remove overlapping elements, keeping the one with highest confidence"""
        if not elements:
            return elements
        
        # Sort by confidence (highest first)
        elements.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_elements = []
        
        for element in elements:
            # Check if this element overlaps with any already accepted element
            overlaps = False
            for accepted in filtered_elements:
                if element.bounding_box.overlaps_with(accepted.bounding_box):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_elements.append(element)
        
        return filtered_elements
    
    def _detect_with_ocr(self, image_path: str) -> List[DetectedElement]:
        """Detect text elements using OCR"""
        elements = []
        
        if not self.ocr_reader:
            self.logger.warning("OCR not available")
            return elements
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Perform OCR
            if hasattr(self.ocr_reader, 'readtext'):  # EasyOCR
                results = self.ocr_reader.readtext(image)
                
                for (bbox_coords, text, confidence) in results:
                    if confidence >= self.confidence_threshold:
                        # Convert bbox coordinates
                        x_coords = [coord[0] for coord in bbox_coords]
                        y_coords = [coord[1] for coord in bbox_coords]
                        
                        bbox = BoundingBox(
                            x=int(min(x_coords)),
                            y=int(min(y_coords)),
                            width=int(max(x_coords) - min(x_coords)),
                            height=int(max(y_coords) - min(y_coords))
                        )
                        
                        element = DetectedElement(
                            element_type=ElementType.UNKNOWN,
                            confidence=confidence,
                            bounding_box=bbox,
                            text=text
                        )
                        
                        elements.append(element)
            
            else:  # Tesseract OCR
                # Use pytesseract image_to_data for bounding box info
                import pytesseract
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    confidence = data['conf'][i] / 100.0  # Convert to 0-1 range
                    
                    if text and confidence >= self.confidence_threshold:
                        bbox = BoundingBox(
                            x=data['left'][i],
                            y=data['top'][i],
                            width=data['width'][i],
                            height=data['height'][i]
                        )
                        
                        element = DetectedElement(
                            element_type=ElementType.UNKNOWN,
                            confidence=confidence,
                            bounding_box=bbox,
                            text=text
                        )
                        
                        elements.append(element)
            
            self.logger.info(f"OCR found {len(elements)} text elements")
            return elements
            
        except Exception as e:
            self.logger.error(f"OCR detection failed: {e}")
            return []
    
    # =================== ELEMENT SEARCH ===================
    
    def find_element_by_text(self, text: str, image_path: Optional[str] = None) -> Optional[DetectedElement]:
        """
        Find element containing specific text
        
        Args:
            text: Text to search for
            image_path: Optional image path, if None takes new screenshot
            
        Returns:
            DetectedElement or None if not found
        """
        try:
            if image_path is None:
                image_path = self.take_screenshot()
            
            # Use OCR to find text elements
            elements = self._detect_with_ocr(image_path)
            
            # Search for matching text
            for element in elements:
                if element.text and text.lower() in element.text.lower():
                    return element
            
            return None
            
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return None
    
    def find_elements_by_type(self, element_type: ElementType, image_path: Optional[str] = None) -> List[DetectedElement]:
        """
        Find all elements of a specific type
        
        Args:
            element_type: Type of elements to find
            image_path: Optional image path, if None takes new screenshot
            
        Returns:
            List of matching elements
        """
        try:
            if image_path is None:
                image_path = self.take_screenshot()
            
            # Use template matching to find elements
            elements = self._detect_with_template_matching(image_path)
            
            # Filter by type
            matching_elements = [e for e in elements if e.element_type == element_type]
            
            return matching_elements
            
        except Exception as e:
            self.logger.error(f"Type search failed: {e}")
            return []
    
    def find_element_at_position(self, x: int, y: int, image_path: Optional[str] = None) -> Optional[DetectedElement]:
        """
        Find element at specific screen position
        
        Args:
            x: X coordinate
            y: Y coordinate
            image_path: Optional image path, if None takes new screenshot
            
        Returns:
            DetectedElement or None if not found
        """
        try:
            if image_path is None:
                image_path = self.take_screenshot()
            
            # Get all elements
            elements = self.detect_elements(image_path)
            
            # Find element containing the point
            for element in elements:
                bbox = element.bounding_box
                if (bbox.x <= x <= bbox.x + bbox.width and
                    bbox.y <= y <= bbox.y + bbox.height):
                    return element
            
            return None
            
        except Exception as e:
            self.logger.error(f"Position search failed: {e}")
            return None
    
    # =================== VISUALIZATION ===================
    
    def visualize_detections(self, analysis: ScreenAnalysis, output_path: Optional[str] = None) -> str:
        """
        Create visualization of detected elements
        
        Args:
            analysis: Screen analysis results
            output_path: Optional output path
            
        Returns:
            str: Path to visualization image
        """
        try:
            if not Image:
                raise ImportError("PIL not available")
            
            # Load screenshot
            image = Image.open(analysis.screenshot_path)
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Color mapping for element types
            colors = {
                ElementType.BUTTON: "red",
                ElementType.TEXT_FIELD: "blue",
                ElementType.DROPDOWN: "green",
                ElementType.CHECKBOX: "orange",
                ElementType.RADIO_BUTTON: "purple",
                ElementType.LINK: "cyan",
                ElementType.IMAGE: "yellow",
                ElementType.ICON: "magenta",
                ElementType.WINDOW: "brown",
                ElementType.MENU: "pink",
                ElementType.UNKNOWN: "gray"
            }
            
            # Draw bounding boxes and labels
            for i, element in enumerate(analysis.elements):
                bbox = element.bounding_box
                color = colors.get(element.element_type, "gray")
                
                # Draw bounding box
                draw.rectangle([
                    bbox.x, bbox.y,
                    bbox.x + bbox.width, bbox.y + bbox.height
                ], outline=color, width=2)
                
                # Create label
                label_parts = [f"{i+1}"]
                if element.template_name:
                    label_parts.append(element.template_name)
                if element.text:
                    label_parts.append(f'"{element.text[:20]}"')
                label_parts.append(f"{element.confidence:.2f}")
                
                label = " ".join(label_parts)
                
                # Draw label background
                label_bbox = draw.textbbox((bbox.x, bbox.y - 20), label, font=font)
                draw.rectangle(label_bbox, fill=color)
                
                # Draw label text
                draw.text((bbox.x, bbox.y - 20), label, fill="white", font=font)
            
            # Save visualization
            if output_path is None:
                timestamp = int(time.time())
                output_path = str(self.screenshots_path / f"visualization_{timestamp}.png")
            
            image.save(output_path)
            self.logger.info(f"Visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise
    
    # =================== UTILITY METHODS ===================
    
    def save_element_as_template(self, element: DetectedElement, screenshot_path: str, template_name: str):
        """
        Save detected element as a template for future matching
        
        Args:
            element: Detected element to save
            screenshot_path: Path to screenshot containing the element
            template_name: Name for the new template
        """
        try:
            # Load screenshot
            image = cv2.imread(screenshot_path)
            if image is None:
                raise ValueError(f"Could not load screenshot: {screenshot_path}")
            
            # Extract element region
            bbox = element.bounding_box
            element_image = image[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
            
            # Save template
            template_path = self.templates_path / f"{template_name}.png"
            cv2.imwrite(str(template_path), element_image)
            
            # Update mapping if element type is known
            if element.element_type != ElementType.UNKNOWN:
                self.template_type_mapping[template_name] = element.element_type
                self._save_template_mapping()
            
            self.logger.info(f"Template saved: {template_path}")
            
        except Exception as e:
            self.logger.error(f"Template saving failed: {e}")
            raise
    
    def _save_template_mapping(self):
        """Save template to element type mapping"""
        try:
            mapping_file = self.templates_path / "element_mapping.json"
            mapping_data = {k: v.value for k, v in self.template_type_mapping.items()}
            
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save template mapping: {e}")
    
    def get_element_info(self, element: DetectedElement) -> Dict[str, Any]:
        """Get detailed information about an element"""
        return {
            "type": element.element_type.value,
            "confidence": element.confidence,
            "position": {
                "x": element.bounding_box.x,
                "y": element.bounding_box.y,
                "width": element.bounding_box.width,
                "height": element.bounding_box.height,
                "center": element.bounding_box.center
            },
            "text": element.text,
            "template_name": element.template_name,
            "attributes": element.attributes
        }
    
    def export_analysis(self, analysis: ScreenAnalysis, output_path: str):
        """Export analysis results to JSON file"""
        try:
            data = {
                "screenshot_path": analysis.screenshot_path,
                "timestamp": analysis.timestamp,
                "screen_size": analysis.screen_size,
                "analysis_duration": analysis.analysis_duration,
                "elements": [self.get_element_info(element) for element in analysis.elements]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Analysis exported: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Analysis export failed: {e}")
            raise


# =================== EXAMPLE USAGE ===================

def main():
    """Example usage of VisionService"""
    
    # Initialize service
    config = {
        'confidence_threshold': 0.7,
        'ocr_enabled': True,
        'default_model': 'template'
    }
    
    vision = VisionService(config)
    
    try:
        # Analyze current screen
        print("Analyzing screen...")
        analysis = vision.analyze_screen()
        
        print(f"Found {len(analysis.elements)} elements in {analysis.analysis_duration:.2f}s")
        
        # Print element details
        for i, element in enumerate(analysis.elements):
            info = vision.get_element_info(element)
            print(f"Element {i+1}: {info}")
        
        # Create visualization
        viz_path = vision.visualize_detections(analysis)
        print(f"Visualization saved: {viz_path}")
        
        # Export analysis
        export_path = "screen_analysis.json"
        vision.export_analysis(analysis, export_path)
        print(f"Analysis exported: {export_path}")
        
        # Search for elements
        buttons = vision.find_elements_by_type(ElementType.BUTTON)
        print(f"Found {len(buttons)} buttons")
        
        # Search for text
        element = vision.find_element_by_text("OK")
        if element:
            print(f"Found 'OK' element at {element.bounding_box.center}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()