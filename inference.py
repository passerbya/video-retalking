import numpy as np
import cv2, os, sys, subprocess, platform, torch
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.append('third_part')
# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.face_detect.retinaface_detection import RetinaFaceDetection
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, merge_face, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings
warnings.filterwarnings("ignore")
import hashlib
import shutil

args = options()

def md5sum(f):
    m = hashlib.md5()
    n = 1024 * 8
    inp = open(f, 'rb')
    try:
        while True:
            buf = inp.read(n)
            if not buf:
                break
            m.update(buf)
    finally:
        inp.close()

    return m.hexdigest()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    base_name = md5sum(args.face)
    tmp_dir = '{}/{}'.format(args.tmp_dir, base_name)
    os.makedirs(tmp_dir, exist_ok=True)
    temp_audio = '{}/temp.wav'.format(tmp_dir)
    if not args.audio.endswith('.wav'):
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(args.audio, temp_audio)
        subprocess.call(command, shell=True)
        args.audio = temp_audio
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        _, frame = video_stream.read()
        height, width = frame.shape[:-1]
        video_stream.release()

    mel_step_size, mel_idx_multiplier, i, mel_chunks = 16, 80./fps, 0, []
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Load audio; Length of mel chunks: {}".format(len(mel_chunks)))

    global enhancer, restorer, croper, net_recon, lm3d_std, expression, D_Net, model
    enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.4.pth', upscale=1)
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    net_recon = load_face3d_net(args.face3d_net_path, device)
    lm3d_std = load_lm3d('checkpoints/BFM')

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from',args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')

        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil])[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
    elif args.exp_img == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using expression center')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(args, device)

    print("Load model done.")

    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        height, width = full_frames[0].shape[:-1]
        temp_video = '{}/result.mp4'.format(tmp_dir)
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        lipsync(out, args, device, fps, mel_chunks, full_frames)
    else:
        video_stream = cv2.VideoCapture(args.face)
        args.pads = np.int32(np.array(args.pads) * max(height/480, 1))

        facedetector = RetinaFaceDetection('checkpoints', device)
        temp_video = '{}/result.mp4'.format(tmp_dir)
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        index = 0

        while True:
            frames = read_frames(video_stream, args.frame_batch_size)
            if len(frames) == 0:
                break

            _face_indexes = []
            _full_frames = []
            _mel_batch = []
            for idx in range(len(frames)):
                img_512 = np.array(cv2.resize(frames[idx],(512,512)))
                facebs, landms = facedetector.detect(img_512)
                #print(len(facebs), len(landms))

                if len(facebs) == 0 and len(landms) == 0:
                    #no face
                    if len(_face_indexes) == 0 or _face_indexes[-1]:
                        _face_indexes.append(False)
                        _full_frames.append([])
                        _mel_batch.append([])
                else:
                    if len(_face_indexes) == 0 or not _face_indexes[-1]:
                        _face_indexes.append(True)
                        _full_frames.append([])
                        _mel_batch.append([])

                _full_frames[-1].append(frames[idx])
                if index < len(mel_chunks):
                    _mel_batch[-1].append(mel_chunks[index])
                index += 1

            for idx in range(len(_face_indexes)):
                full_frames = _full_frames[idx]
                mel_batch = _mel_batch[idx]

                if not _face_indexes[idx]:
                    print(len(full_frames), 'frames has no face')
                    for frame in full_frames:
                        out.write(frame)
                else:
                    if len(full_frames) < 2:
                        print('1 frame has face')
                        for frame in full_frames:
                            out.write(frame)
                    else:
                        print(len(full_frames), 'frames has face')
                        lipsync(out, args, device, fps, mel_batch, full_frames)

        video_stream.release()

    out.release()

    if not os.path.isdir(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    command = 'ffmpeg -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, temp_video, args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    print('outfile:', args.outfile)

def lipsync(out, args, device, fps, mel_chunks, full_frames):
    global enhancer, restorer, croper, net_recon, lm3d_std, expression, D_Net, model
    frame_h, frame_w = full_frames[0].shape[:-1]
    print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    crop_result = croper.crop(full_frames_RGB, xsize=512)
    if crop_result is None:
        #no face
        print(len(full_frames), 'frames has no face')
        for frame in full_frames:
            out.write(frame)
        return
    full_frames_RGB, crop, quad = crop_result

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, frame_h), clx+lx, min(clx+rx, frame_w)
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    print('[Step 1] Landmarks Extraction in Video.')
    kp_extractor = KeypointExtractor()
    lm = kp_extractor.extract_keypoint(frames_pil)

    video_coeffs = []
    for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
        frame = frames_pil[idx]
        W, H = frame.size
        lm_idx = lm[idx].reshape([-1, 2])
        if np.mean(lm_idx) == -1:
            lm_idx = (lm3d_std[:, :2]+1) / 2.
            lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
        else:
            lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

        trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        with torch.no_grad():
            coeffs = split_coeff(net_recon(im_idx_tensor))

        pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
        pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'], \
                                     pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
        video_coeffs.append(pred_coeff)
    semantic_npy = np.array(video_coeffs)[:,0]

    imgs = []
    for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stablize the expression In Video:"):
        if args.one_shot:
            source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
            semantic_source_numpy = semantic_npy[0:1]
        else:
            source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
            semantic_source_numpy = semantic_npy[idx:idx+1]
        ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
        coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)

        # hacking the new expression
        coeff[:, :64, :] = expression[None, :64, None].to(device)
        with torch.no_grad():
            output = D_Net(source_img, coeff)
        img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
        imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR))
    torch.cuda.empty_cache()

    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 4] Reference Enhancement'):
        img = imgs[idx]
        try:
            pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
            imgs_enhanced.append(pred)
        except UnboundLocalError:
            #no face
            print('frames has no face')
            imgs_enhanced.append(img)

    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))

    #ii = 0
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 5] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB

        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1)
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
            else:
                pass

            if args.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'),
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(instance.fake_img / 2. + 0.5, size=(384, 384), mode='bilinear')

            if args.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete))
                pred = pred * mask + cur_gen_faces * (1 - mask)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        torch.cuda.empty_cache()
        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            if c[0]==0 and c[1]==0 and c[2]==0 and c[3]==0:
                out.write(xf)
            else:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                yy1, yy2, xx1, xx2 = merge_face(xf, c)
                ff = xf.copy()[yy1: yy2, xx1: xx2]
                ff[y1-yy1: y2-yy1, x1-xx1: x2-xx1] = p
                # month region enhancement by GFPGAN
                _, _, restored_img = restorer.enhance(ff, has_aligned=False, only_center_face=True, paste_back=True)
                '''
                0: 'background' 1: 'skin'   2: 'nose'
                3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
                6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
                9: 'r_ear'  10: 'mouth' 11: 'u_lip'
                12: 'l_lip' 13: 'hair'  14: 'hat'
                15: 'ear_r' 16: 'neck_l'    17: 'neck'
                18: 'cloth'
                '''
                mm = [255, 255, 255, 255, 255, 255,255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
                mouse_mask = np.zeros_like(restored_img)
                enhancer.faceparser.size = yy2 - yy1
                tmp_mask = enhancer.faceparser.process(restored_img[y1-yy1: y2-yy1, x1-xx1: x2-xx1], mm)[0]
                mouse_mask[y1-yy1: y2-yy1, x1-xx1: x2-xx1] = cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.
                full_mask = np.float32(mouse_mask)
                '''
                cv2.imwrite(f"face/{ii:05d}_p.jpg", p)
                cv2.imwrite(f"face/{ii:05d}_ff.jpg", ff)
                cv2.imwrite(f"face/{ii:05d}_restored_img.jpg", restored_img)
                '''
                #print(np.shape(restored_img), np.shape(ff), np.shape(full_mask[:, :, 0]))
                img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
                pp = np.uint8(np.clip(img, 0 ,255))
                #cv2.imwrite(f"face/{ii:05d}_pp1.jpg", pp)
                pf = xf.copy()
                pf[yy1: yy2, xx1: xx2] = pp
                try:
                    pp, orig_faces, enhanced_faces = enhancer.process(pf, xf, bbox=c, face_enhance=False, possion_blending=True)
                    #cv2.imwrite(f"face/{ii:05d}_pp2.jpg", pp)
                    xf[yy1: yy2, xx1: xx2] = pp[yy1: yy2, xx1: xx2]
                    #cv2.imwrite(f"face/{ii:05d}_xf.jpg", xf)
                    out.write(xf)
                except UnboundLocalError:
                    #no face
                    out.write(xf)
                #ii += 1

def read_frames(video_stream, frame_size):
    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break
        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)
        if len(full_frames) >= frame_size:
            return full_frames
    return full_frames

# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    refs = []
    image_size = 256 

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil)
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    inverse_transforms = [calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for quad in quads]
    del kp_extractor.detector

    oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        if oface.shape[0] == 0 and oface.shape[1] == 0:
            #print('oface', coords, oface, np.shape(full_frames[idx]))
            coords = [0, 0, 0, 0]

        if coords[0]==0 and coords[1]==0 and coords[2]==0 and coords[3]==0:
            face = full_frames[idx][0:args.img_size, 0:args.img_size]
            oface = full_frames[idx][0:args.img_size, 0:args.img_size]
        else:
            face = cv2.resize(face, (args.img_size, args.img_size))
            oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch

enhancer = None
restorer = None
croper = None
net_recon = None
lm3d_std = None
expression = None
D_Net = None
model = None
if __name__ == '__main__':
    main()
