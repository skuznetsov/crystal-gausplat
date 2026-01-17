// CoreML/Vision bridge for Crystal FFI
// Provides YOLO object detection on Apple Silicon

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#import <Accelerate/Accelerate.h>

// ============================================================================
// Detection result structure
// ============================================================================

typedef struct {
    float x;        // center x (normalized 0-1)
    float y;        // center y (normalized 0-1)
    float width;    // width (normalized 0-1)
    float height;   // height (normalized 0-1)
    float confidence;
    int class_id;
    int track_id;   // -1 if not tracked
} GSDetection;

typedef struct {
    GSDetection* detections;
    int count;
    int capacity;
} GSDetectionArray;

// ============================================================================
// CoreML Model wrapper
// ============================================================================

static MLModel* g_yolo_model = nil;
static VNCoreMLModel* g_vision_model = nil;
static NSArray<NSString*>* g_class_names = nil;

// Forward declaration
static NSArray<VNRecognizedObjectObservation*>* applyNMS(
    NSArray<VNRecognizedObjectObservation*>* detections,
    float threshold
);

// Load YOLO model from .mlmodelc path
extern "C" int gs_coreml_load_yolo(const char* model_path) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:model_path];
        NSURL* url = [NSURL fileURLWithPath:path];

        NSError* error = nil;

        // Compile model if needed (for .mlmodel files)
        if ([path hasSuffix:@".mlmodel"]) {
            NSURL* compiledURL = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                NSLog(@"CoreML compile error: %@", error);
                return -1;
            }
            url = compiledURL;
        }

        // Load model
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll; // Use Neural Engine + GPU

        g_yolo_model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (error) {
            NSLog(@"CoreML load error: %@", error);
            return -2;
        }

        // Create Vision model
        g_vision_model = [VNCoreMLModel modelForMLModel:g_yolo_model error:&error];
        if (error) {
            NSLog(@"Vision model error: %@", error);
            return -3;
        }

        // Default COCO class names
        g_class_names = @[
            @"person", @"bicycle", @"car", @"motorcycle", @"airplane", @"bus", @"train", @"truck",
            @"boat", @"traffic light", @"fire hydrant", @"stop sign", @"parking meter", @"bench",
            @"bird", @"cat", @"dog", @"horse", @"sheep", @"cow", @"elephant", @"bear", @"zebra",
            @"giraffe", @"backpack", @"umbrella", @"handbag", @"tie", @"suitcase", @"frisbee",
            @"skis", @"snowboard", @"sports ball", @"kite", @"baseball bat", @"baseball glove",
            @"skateboard", @"surfboard", @"tennis racket", @"bottle", @"wine glass", @"cup",
            @"fork", @"knife", @"spoon", @"bowl", @"banana", @"apple", @"sandwich", @"orange",
            @"broccoli", @"carrot", @"hot dog", @"pizza", @"donut", @"cake", @"chair", @"couch",
            @"potted plant", @"bed", @"dining table", @"toilet", @"tv", @"laptop", @"mouse",
            @"remote", @"keyboard", @"cell phone", @"microwave", @"oven", @"toaster", @"sink",
            @"refrigerator", @"book", @"clock", @"vase", @"scissors", @"teddy bear", @"hair drier",
            @"toothbrush"
        ];

        NSLog(@"CoreML YOLO model loaded successfully");
        return 0;
    }
}

// Run detection on RGB image data
extern "C" int gs_coreml_detect(
    const uint8_t* rgb_data,
    int width,
    int height,
    float confidence_threshold,
    float nms_threshold,
    GSDetectionArray* out_detections
) {
    @autoreleasepool {
        if (!g_vision_model) {
            return -1;
        }

        // Create CGImage from RGB data
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            (void*)rgb_data,
            width, height,
            8, width * 3,
            colorSpace,
            kCGImageAlphaNone | kCGBitmapByteOrderDefault
        );
        CGImageRef cgImage = CGBitmapContextCreateImage(context);
        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);

        if (!cgImage) {
            return -2;
        }

        // Create Vision request
        __block NSArray<VNRecognizedObjectObservation*>* observations = nil;
        __block NSError* requestError = nil;

        VNCoreMLRequest* request = [[VNCoreMLRequest alloc] initWithModel:g_vision_model
            completionHandler:^(VNRequest* req, NSError* error) {
                if (error) {
                    requestError = error;
                    return;
                }
                observations = req.results;
            }];

        request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;

        // Run request
        VNImageRequestHandler* handler = [[VNImageRequestHandler alloc]
            initWithCGImage:cgImage options:@{}];

        NSError* error = nil;
        [handler performRequests:@[request] error:&error];

        CGImageRelease(cgImage);

        if (error || requestError) {
            NSLog(@"Detection error: %@", error ?: requestError);
            return -3;
        }

        // Parse results
        NSMutableArray<VNRecognizedObjectObservation*>* filtered = [NSMutableArray array];

        for (VNRecognizedObjectObservation* obs in observations) {
            if (obs.confidence >= confidence_threshold) {
                [filtered addObject:obs];
            }
        }

        // Apply NMS
        NSArray<VNRecognizedObjectObservation*>* nmsResults = applyNMS(filtered, nms_threshold);

        // Convert to output format
        int count = (int)nmsResults.count;
        if (count > out_detections->capacity) {
            count = out_detections->capacity;
        }

        for (int i = 0; i < count; i++) {
            VNRecognizedObjectObservation* obs = nmsResults[i];
            CGRect bbox = obs.boundingBox;

            // Vision uses bottom-left origin, convert to center-based
            out_detections->detections[i].x = bbox.origin.x + bbox.size.width / 2;
            out_detections->detections[i].y = 1.0 - (bbox.origin.y + bbox.size.height / 2); // Flip Y
            out_detections->detections[i].width = bbox.size.width;
            out_detections->detections[i].height = bbox.size.height;
            out_detections->detections[i].confidence = obs.confidence;

            // Get class ID from label
            VNClassificationObservation* topLabel = obs.labels.firstObject;
            int classId = 0;
            if (topLabel) {
                NSUInteger idx = [g_class_names indexOfObject:topLabel.identifier];
                if (idx != NSNotFound) {
                    classId = (int)idx;
                }
            }
            out_detections->detections[i].class_id = classId;
            out_detections->detections[i].track_id = -1;
        }

        out_detections->count = count;
        return 0;
    }
}

// NMS helper
static NSArray<VNRecognizedObjectObservation*>* applyNMS(
    NSArray<VNRecognizedObjectObservation*>* detections,
    float threshold
) {
    if (detections.count == 0) return @[];

    // Sort by confidence
    NSArray* sorted = [detections sortedArrayUsingComparator:^NSComparisonResult(
        VNRecognizedObjectObservation* a, VNRecognizedObjectObservation* b) {
        return b.confidence > a.confidence ? NSOrderedDescending : NSOrderedAscending;
    }];

    NSMutableArray* kept = [NSMutableArray array];
    NSMutableIndexSet* suppressed = [NSMutableIndexSet indexSet];

    for (NSUInteger i = 0; i < sorted.count; i++) {
        if ([suppressed containsIndex:i]) continue;

        VNRecognizedObjectObservation* obs = sorted[i];
        [kept addObject:obs];

        CGRect boxA = obs.boundingBox;

        for (NSUInteger j = i + 1; j < sorted.count; j++) {
            if ([suppressed containsIndex:j]) continue;

            CGRect boxB = ((VNRecognizedObjectObservation*)sorted[j]).boundingBox;

            // Compute IoU
            CGRect intersection = CGRectIntersection(boxA, boxB);
            if (CGRectIsNull(intersection)) continue;

            float intersectionArea = intersection.size.width * intersection.size.height;
            float unionArea = boxA.size.width * boxA.size.height +
                             boxB.size.width * boxB.size.height - intersectionArea;

            float iou = intersectionArea / unionArea;

            if (iou > threshold) {
                [suppressed addIndex:j];
            }
        }
    }

    return kept;
}

// Get class name by ID
extern "C" const char* gs_coreml_class_name(int class_id) {
    if (!g_class_names || class_id < 0 || class_id >= (int)g_class_names.count) {
        return "unknown";
    }
    return [g_class_names[class_id] UTF8String];
}

// Check if model is loaded
extern "C" int gs_coreml_is_loaded(void) {
    return g_vision_model != nil ? 1 : 0;
}

// Release model
extern "C" void gs_coreml_release(void) {
    g_yolo_model = nil;
    g_vision_model = nil;
}

// ============================================================================
// Kalman Filter for tracking (simple 2D position + velocity)
// ============================================================================

typedef struct {
    float x, y;       // position
    float vx, vy;     // velocity
    float w, h;       // size
    float P[16];      // 4x4 covariance matrix (flattened)
    int id;
    int hits;
    int misses;
    int class_id;
} GSKalmanTrack;

// Process noise
static const float Q_POS = 1.0f;
static const float Q_VEL = 0.1f;

// Measurement noise
static const float R_MEAS = 1.0f;

extern "C" void gs_kalman_predict(GSKalmanTrack* track, float dt) {
    // State transition: x' = x + vx*dt, y' = y + vy*dt
    track->x += track->vx * dt;
    track->y += track->vy * dt;

    // Update covariance P = F*P*F' + Q
    // Simplified: just increase uncertainty
    track->P[0] += Q_POS * dt * dt;  // x variance
    track->P[5] += Q_POS * dt * dt;  // y variance
    track->P[10] += Q_VEL * dt;      // vx variance
    track->P[15] += Q_VEL * dt;      // vy variance
}

extern "C" void gs_kalman_update(GSKalmanTrack* track, float mx, float my, float mw, float mh) {
    // Kalman gain (simplified)
    float k = track->P[0] / (track->P[0] + R_MEAS);

    // Update state
    float dx = mx - track->x;
    float dy = my - track->y;

    track->x += k * dx;
    track->y += k * dy;
    track->vx = 0.8f * track->vx + 0.2f * dx;  // Smooth velocity update
    track->vy = 0.8f * track->vy + 0.2f * dy;
    track->w = 0.7f * track->w + 0.3f * mw;
    track->h = 0.7f * track->h + 0.3f * mh;

    // Update covariance
    track->P[0] *= (1.0f - k);
    track->P[5] *= (1.0f - k);

    track->hits++;
    track->misses = 0;
}

extern "C" void gs_kalman_init(GSKalmanTrack* track, int id, float x, float y, float w, float h, int class_id) {
    track->x = x;
    track->y = y;
    track->vx = 0;
    track->vy = 0;
    track->w = w;
    track->h = h;
    track->id = id;
    track->hits = 1;
    track->misses = 0;
    track->class_id = class_id;

    // Initialize covariance
    memset(track->P, 0, sizeof(track->P));
    track->P[0] = 10.0f;   // x
    track->P[5] = 10.0f;   // y
    track->P[10] = 10.0f;  // vx
    track->P[15] = 10.0f;  // vy
}

// ============================================================================
// Hungarian Algorithm for optimal assignment
// ============================================================================

// Simple greedy assignment (faster, good enough for small N)
extern "C" void gs_hungarian_assign(
    const float* cost_matrix,  // N_tracks x M_detections
    int n_tracks,
    int m_detections,
    int* assignments,          // output: track_i -> detection_j (-1 if unassigned)
    float max_cost
) {
    // Initialize all to unassigned
    for (int i = 0; i < n_tracks; i++) {
        assignments[i] = -1;
    }

    if (n_tracks == 0 || m_detections == 0) return;

    // Track which detections are taken
    bool* det_taken = (bool*)calloc(m_detections, sizeof(bool));

    // Greedy: for each track, find best available detection
    for (int iter = 0; iter < n_tracks; iter++) {
        float best_cost = max_cost;
        int best_track = -1;
        int best_det = -1;

        for (int i = 0; i < n_tracks; i++) {
            if (assignments[i] >= 0) continue;  // Already assigned

            for (int j = 0; j < m_detections; j++) {
                if (det_taken[j]) continue;

                float cost = cost_matrix[i * m_detections + j];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_track = i;
                    best_det = j;
                }
            }
        }

        if (best_track >= 0 && best_det >= 0) {
            assignments[best_track] = best_det;
            det_taken[best_det] = true;
        }
    }

    free(det_taken);
}

// Compute IoU between track and detection
extern "C" float gs_compute_iou(
    float tx, float ty, float tw, float th,  // track (center, size)
    float dx, float dy, float dw, float dh   // detection (center, size)
) {
    // Convert to corners
    float t_x1 = tx - tw/2, t_y1 = ty - th/2;
    float t_x2 = tx + tw/2, t_y2 = ty + th/2;
    float d_x1 = dx - dw/2, d_y1 = dy - dh/2;
    float d_x2 = dx + dw/2, d_y2 = dy + dh/2;

    // Intersection
    float ix1 = fmax(t_x1, d_x1);
    float iy1 = fmax(t_y1, d_y1);
    float ix2 = fmin(t_x2, d_x2);
    float iy2 = fmin(t_y2, d_y2);

    float iw = fmax(0.0f, ix2 - ix1);
    float ih = fmax(0.0f, iy2 - iy1);
    float intersection = iw * ih;

    // Union
    float t_area = tw * th;
    float d_area = dw * dh;
    float union_area = t_area + d_area - intersection;

    if (union_area <= 0) return 0.0f;
    return intersection / union_area;
}
