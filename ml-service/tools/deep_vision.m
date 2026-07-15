#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

static VNFeaturePrintObservation *FeaturePrintForPath(NSString *path, NSError **error) {
    NSURL *url = [NSURL fileURLWithPath:path];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    if (source == NULL) {
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"TurtleDeepVision"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Unable to open image: %@", path]}];
        }
        return nil;
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
    CFRelease(source);
    if (image == NULL) {
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"TurtleDeepVision"
                                         code:2
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Unable to decode image: %@", path]}];
        }
        return nil;
    }

    VNGenerateImageFeaturePrintRequest *request = [[VNGenerateImageFeaturePrintRequest alloc] init];
    request.revision = VNGenerateImageFeaturePrintRequestRevision2;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
    CGImageRelease(image);
    if (![handler performRequests:@[request] error:error]) {
        return nil;
    }
    return request.results.firstObject;
}

static NSString *Timestamp(void) {
    NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
    formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
    return [formatter stringFromDate:[NSDate date]];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "Usage: deep_vision <image-manifest.json> <output.json>\n");
            return 2;
        }

        NSString *manifestPath = [NSString stringWithUTF8String:argv[1]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[2]];
        NSError *error = nil;
        NSData *manifestData = [NSData dataWithContentsOfFile:manifestPath options:0 error:&error];
        if (manifestData == nil) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        NSArray<NSDictionary *> *manifest = [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&error];
        if (![manifest isKindOfClass:[NSArray class]]) {
            fprintf(stderr, "Invalid image manifest: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        NSMutableDictionary<NSString *, VNFeaturePrintObservation *> *observations = [NSMutableDictionary dictionary];
        NSMutableDictionary<NSString *, NSString *> *groups = [NSMutableDictionary dictionary];
        NSMutableArray<NSDictionary *> *failures = [NSMutableArray array];

        [manifest enumerateObjectsUsingBlock:^(NSDictionary *entry, NSUInteger index, BOOL *stop) {
            NSString *key = entry[@"key"];
            NSString *path = entry[@"path"];
            NSString *group = entry[@"group"];
            NSError *featureError = nil;
            VNFeaturePrintObservation *observation = FeaturePrintForPath(path, &featureError);
            if (observation != nil) {
                observations[key] = observation;
                groups[key] = group;
            } else {
                [failures addObject:@{
                    @"id": key ?: @"unknown",
                    @"reason": featureError.localizedDescription ?: @"Unknown Vision error"
                }];
            }
            if ((index + 1) % 25 == 0 || index + 1 == manifest.count) {
                fprintf(stderr, "Embedded %lu/%lu images\n", (unsigned long)index + 1, (unsigned long)manifest.count);
            }
        }];

        NSArray<NSString *> *allIds = [[observations allKeys] sortedArrayUsingSelector:@selector(compare:)];
        NSPredicate *pastPredicate = [NSPredicate predicateWithBlock:^BOOL(NSString *key, NSDictionary *bindings) {
            return [groups[key] isEqualToString:@"past"];
        }];
        NSPredicate *upcomingPredicate = [NSPredicate predicateWithBlock:^BOOL(NSString *key, NSDictionary *bindings) {
            return [groups[key] isEqualToString:@"upcoming"];
        }];
        NSArray<NSString *> *historicalIds = [allIds filteredArrayUsingPredicate:pastPredicate];
        NSArray<NSString *> *upcomingIds = [allIds filteredArrayUsingPredicate:upcomingPredicate];
        NSMutableArray<NSDictionary *> *distances = [NSMutableArray arrayWithCapacity:historicalIds.count * (historicalIds.count + upcomingIds.count)];

        void (^appendDistance)(NSString *, NSString *) = ^(NSString *leftId, NSString *rightId) {
            float distance = 0.0f;
            NSError *distanceError = nil;
            BOOL ok = [observations[leftId] computeDistance:&distance
                                 toFeaturePrintObservation:observations[rightId]
                                                     error:&distanceError];
            if (ok) {
                [distances addObject:@{
                    @"leftId": leftId,
                    @"rightId": rightId,
                    @"distance": @(distance)
                }];
            } else {
                [failures addObject:@{
                    @"id": [NSString stringWithFormat:@"%@:%@", leftId, rightId],
                    @"reason": distanceError.localizedDescription ?: @"Distance calculation failed"
                }];
            }
        };

        for (NSString *leftId in historicalIds) {
            for (NSString *rightId in historicalIds) {
                appendDistance(leftId, rightId);
            }
        }
        for (NSString *leftId in upcomingIds) {
            for (NSString *rightId in historicalIds) {
                appendDistance(leftId, rightId);
            }
        }

        NSDictionary *output = @{
            @"engine": @"Apple Vision FeaturePrint v2 (deep neural image embedding)",
            @"revision": @(VNGenerateImageFeaturePrintRequestRevision2),
            @"generatedAt": Timestamp(),
            @"successfulImages": @(observations.count),
            @"failures": failures,
            @"distances": distances
        };
        NSData *outputData = [NSJSONSerialization dataWithJSONObject:output options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
        if (outputData == nil || ![outputData writeToFile:outputPath options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fprintf(stderr, "Wrote %lu deep-vision distances to %s\n", (unsigned long)distances.count, outputPath.UTF8String);
    }
    return 0;
}
