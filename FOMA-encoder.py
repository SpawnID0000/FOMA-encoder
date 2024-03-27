import subprocess
import os
import sys
import librosa
import shutil
import soundfile as sf
import numpy as np

def get_audio_properties(input_file_path):
    cmd_info = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,duration,bits_per_raw_sample',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_file_path,
    ]
    try:
        output = subprocess.check_output(cmd_info).decode().strip().split('\n')
        sample_rate, num_channels, duration, bits_per_raw_sample = output
        bit_depth = bits_per_raw_sample if bits_per_raw_sample.isdigit() else 'Unknown'
        num_samples = int(float(duration) * float(sample_rate))
        file_size = os.path.getsize(input_file_path)
        return int(sample_rate), int(bit_depth), int(num_channels), num_samples, file_size
    except subprocess.CalledProcessError as e:
        print(f"Failed to get audio properties: {e}")
        sys.exit(1)

def extract_and_convert_album_art_to_jpg(input_file_path, output_folder):
    output_jpg_path = os.path.join(output_folder, "cover.jpg")
    cmd_extract = ['ffmpeg', '-i', input_file_path, '-an', '-vcodec', 'copy', '-vframes', '1', output_jpg_path, '-y']
    try:
        subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Album art extracted and saved as {output_jpg_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract album art: {e}")
        return

def resample_audio(input_file_path, output_suffix, target_sample_rate, target_bit_depth, output_folder, base_name, compression_level=8):
    output_file_path = os.path.join(output_folder, f"{base_name}{output_suffix}.flac")
    cmd = [
        'ffmpeg',
        '-i', input_file_path,
        '-ar', str(target_sample_rate),
        '-acodec', 'flac',
        '-compression_level', str(compression_level),
        '-blocksize', '4096',
        output_file_path, '-y'
    ]
    if target_bit_depth == 16:
        cmd.insert(-2, '-sample_fmt')
        cmd.insert(-2, 's16')
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Resampled file created: {output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to resample audio: {e}")

def generate_residual(base_path, target_path, output_residual_path, output_folder):
    base_path = os.path.join(output_folder, base_path)
    target_path = os.path.join(output_folder, target_path)
    output_residual_path = os.path.join(output_folder, output_residual_path)
    base_signal, sr_base = librosa.load(base_path, sr=None, mono=False, dtype='float32')
    target_signal, sr_target = librosa.load(target_path, sr=None, mono=False, dtype='float32')
    if sr_base != sr_target:
        higher_sr = max(sr_base, sr_target)
        if sr_base < higher_sr:
            base_signal = librosa.resample(base_signal, orig_sr=sr_base, target_sr=higher_sr)
        else:
            target_signal = librosa.resample(target_signal, orig_sr=sr_target, target_sr=higher_sr)
        sr_resample = higher_sr
    else:
        sr_resample = sr_base
    min_len = min(len(base_signal), len(target_signal))
    base_signal = base_signal[:min_len]
    target_signal = target_signal[:min_len]
    residual_signal = target_signal - base_signal
    sf.write(output_residual_path, residual_signal.T, sr_resample, format='FLAC', subtype='PCM_24')
    print(f"Residual file created: {output_residual_path}")

def generate_opus_file(input_file_path, output_suffix, sample_rate, bitrate, vbr, frame_duration, application, channels, output_folder, base_name):
    output_file_path = os.path.join(output_folder, f"{base_name}{output_suffix}.opus")
    cmd = [
        'ffmpeg',
        '-i', input_file_path,
        '-ar', str(sample_rate),
        '-acodec', 'libopus',
        '-b:a', bitrate,
        '-vbr', vbr,
        '-frame_duration', str(frame_duration),
        '-application', application,
        '-compression_level', '10',
        '-ac', channels,
        output_file_path,
        '-y'
    ]
    if output_suffix == '_TN' and channels == '1':
            cmd += ['-af', 'volume=-6dB']
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Opus file created: {output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate {output_suffix} Opus file: {e}")

def reconstruct_from_residuals(base_path, residual_paths, reconstructed_path, target_sr, output_folder):
    signal, _ = librosa.load(base_path, sr=target_sr, mono=False, dtype='float32')
    for residual_path in residual_paths:
        residual, _ = librosa.load(residual_path, sr=target_sr, mono=False, dtype='float32')
        min_length = min(signal.shape[1], residual.shape[1])
        signal = signal[:, :min_length] + residual[:, :min_length]
    sf.write(reconstructed_path, signal.T, target_sr, format='FLAC', subtype='PCM_24')
    print(f"Reconstructed file created: {reconstructed_path}")

def reconstruct_and_generate_residuals(input_file_path, output_folder, process_SR, process_HR):
    sample_rate, _, _, _, _ = get_audio_properties(input_file_path)
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    base_path = os.path.join(output_folder, f"{base_name}_CR.flac")
    if process_SR:
        sr_residual_path = os.path.join(output_folder, f"{base_name}_SR_residual.flac")
        sr_reconstructed_path = os.path.join(output_folder, f"{base_name}_SR_reco.flac")
        reconstruct_from_residuals(base_path, [sr_residual_path], sr_reconstructed_path, 48000, output_folder)
        sr_original_path = os.path.join(output_folder, f"{base_name}_SR.flac")
        sr_reco_residual_path = os.path.join(output_folder, f"{base_name}_SR_reco_residual.flac")
        generate_residual(sr_original_path, sr_reconstructed_path, sr_reco_residual_path, output_folder)
    if process_HR:
        hr_residual_path = os.path.join(output_folder, f"{base_name}_HR_residual.flac")
        hr_reconstructed_path = os.path.join(output_folder, f"{base_name}_HR_reco.flac")
        residuals_to_use = [sr_residual_path] if process_SR else []
        residuals_to_use.append(hr_residual_path)
        reconstruct_from_residuals(base_path, residuals_to_use, hr_reconstructed_path, sample_rate, output_folder)
        hr_original_path = os.path.join(output_folder, f"{base_name}_HR.flac")
        hr_reco_residual_path = os.path.join(output_folder, f"{base_name}_HR_reco_residual.flac")
        generate_residual(hr_original_path, hr_reconstructed_path, hr_reco_residual_path, output_folder)

def analyze_residuals_and_delete_if_successful(output_folder, versions, process_SR, process_HR):
    threshold_dbfs = -130
    threshold_linear = 10 ** (threshold_dbfs / 20)
    success = True
    print("Starting analysis of residuals...")
    for version in versions:
        if (version.endswith("SR_reco_residual.flac") and not process_SR) or \
           (version.endswith("HR_reco_residual.flac") and not process_HR):
            continue
        residual_path = os.path.join(output_folder, version)
        print(f"Checking file: {residual_path}")  # Debug line
        try:
            signal, _ = librosa.load(residual_path, sr=None, mono=False, dtype='float32')
            max_abs_value = np.max(np.abs(signal))
            if max_abs_value > threshold_linear:
                success = False
                print(f"Reconstruction verification failed for {version}. Max abs value: {20 * np.log10(max_abs_value)} dBFS")
                break
        except FileNotFoundError:
            print(f"File not found: {residual_path}, skipping.")
            success = False
            break
    if success:
        print("Reconstruction verified successfully for all versions. Proceeding to delete unnecessary files.")
        delete_unnecessary_files(output_folder, process_SR, process_HR)
    else:
        print("Reconstruction verification failed for one or more versions. Retaining all files for manual review.")

def delete_unnecessary_files(output_folder, process_SR, process_HR):
    base_name = os.path.basename(output_folder)
    files_to_delete = []
    if process_SR:
        files_to_delete.extend([
            f"{base_name}_SR.flac", 
            f"{base_name}_SR_reco.flac", 
            f"{base_name}_SR_reco_residual.flac"
        ])
    if process_HR:
        files_to_delete.extend([
            f"{base_name}_HR.flac", 
            f"{base_name}_HR_reco.flac", 
            f"{base_name}_HR_reco_residual.flac"
        ])
    for file_name in files_to_delete:
        file_path = os.path.join(output_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
        else:
            print(f"File not found: {file_path}, skipping deletion.")

def remove_metadata(file_paths):
    for file_path in file_paths:
        if "_TN.opus" in file_path:
            continue
        temp_file_path = file_path.rsplit('.', 1)[0] + '.tmp.' + file_path.rsplit('.', 1)[1]
        cmd = ['ffmpeg', '-i', file_path, '-vn', '-map_metadata', '-1', '-c:a', 'copy', temp_file_path, '-y']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(file_path)
            os.rename(temp_file_path, file_path)
            print(f"Metadata removed from {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove metadata from {file_path}: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

def report_file_sizes(input_file_path, output_folder, process_SR):
    def get_size(start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    def format_size(bytes):
        thresh = 1024
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        for unit in units:
            if bytes < thresh:
                return f"{bytes:.2f} {unit}"
            bytes /= thresh
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    lb_file_path = os.path.join(output_folder, f"{base_name}_LB.opus")
    cr_file_path = os.path.join(output_folder, f"{base_name}_CR.flac")
    input_file_size = os.path.getsize(input_file_path)
    output_folder_size = get_size(output_folder)
    lb_file_size = os.path.getsize(lb_file_path) if os.path.exists(lb_file_path) else 0
    cr_file_size = os.path.getsize(cr_file_path) if os.path.exists(cr_file_path) else 0
    combined_size = input_file_size + lb_file_size + cr_file_size if process_SR else 0
    print(f"Input file size: {format_size(input_file_size)}")
    print(f"LB file size: {format_size(lb_file_size)}")
    print(f"CR file size: {format_size(cr_file_size)}")
    if process_SR:
        print(f"Input + LB + CR combined file size: {format_size(combined_size)}")
    print(f"Total output folder size: {format_size(output_folder_size)}")

#----------------------------------------------------------

def main(input_file_path):

    # Check file format
    file_extension = os.path.splitext(input_file_path)[1].lower()
    if file_extension != '.flac':
        if file_extension in ['.alac', '.aiff', '.wav']:
            print("ERROR: only FLAC input files are supported; please transcode to FLAC and try again.")
            sys.exit(1)
        elif file_extension in ['.mp3', '.aac', '.opus']:
            print("WARNING: lossy files are not valid input for FOMA; please use original (i.e. non-transcoded) lossless files only.")
            sys.exit(1)
        else:
            print(f"ERROR: Unsupported file format {file_extension}; FOMA only supports FLAC files.")
            sys.exit(1)

    # Get audio file properties
    sample_rate, bit_depth, _, _, _ = get_audio_properties(input_file_path)
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    parent_dir = os.path.dirname(input_file_path)

    # Determine processing requirements
    process_SR = sample_rate <= 48000 and bit_depth >= 24 or sample_rate > 48000
    process_HR = sample_rate > 48000 and bit_depth >= 24

    # Create output folder
    output_folder = os.path.join(parent_dir, base_name)
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} already exists. Deleting folder and its contents.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Extract embedded album art
    extract_and_convert_album_art_to_jpg(input_file_path, output_folder)

    # Create FLAC versions
    created_versions = []
    base_path = os.path.splitext(input_file_path)[0]

    if sample_rate > 48000 or bit_depth > 24:
        resample_audio(input_file_path, '_HR', sample_rate, bit_depth, output_folder, base_name)
        created_versions.append('_HR')
        resample_audio(input_file_path, '_SR', 48000, 24, output_folder, base_name)
        created_versions.append('_SR')
    if sample_rate == 48000 or sample_rate == 44100 and bit_depth >= 24:
        if '_SR' not in created_versions:
            resample_audio(input_file_path, '_SR', 48000, 24, output_folder, base_name)
            created_versions.append('_SR')
    resample_audio(input_file_path, '_CR', 48000, 16, output_folder, base_name)
    created_versions.append('_CR')
    #resample_audio(input_file_path, '_CD', 44100, 16, output_folder, base_name)
    #created_versions.append('_CD')

    # Create Opus versions
    generate_opus_file(input_file_path, '_LB', '48000', '128k', 'off', '2.5', 'lowdelay', '2', output_folder, base_name)
    generate_opus_file(input_file_path, '_TN', '12000', '6k', 'on', '40', 'audio', '1', output_folder, base_name)

    # Generate residuals for HR and/or SR if applicable
    print(f"Process SR: {process_SR}")
    print(f"Process HR: {process_HR}")
    if process_HR:
        if '_SR' in created_versions:
            generate_residual(f"{base_name}_SR.flac", f"{base_name}_HR.flac", f"{base_name}_HR_residual.flac", output_folder)
    if process_SR:
        if '_CR' in created_versions:
            generate_residual(f"{base_name}_CR.flac", f"{base_name}_SR.flac", f"{base_name}_SR_residual.flac", output_folder)

    # Generate reconstructed versions and their residuals
    reconstruct_and_generate_residuals(input_file_path, output_folder, process_SR, process_HR)

    # Analyze residuals and delete unneccessary files if successful
    versions_to_analyze = []
    if process_SR:
        versions_to_analyze.append(f"{base_name}_SR_reco_residual.flac")
    if process_HR:
        versions_to_analyze.append(f"{base_name}_HR_reco_residual.flac")
    if versions_to_analyze:
        print(f"Versions to analyze: {versions_to_analyze}")
        analyze_residuals_and_delete_if_successful(output_folder, versions_to_analyze, process_SR, process_HR)
    else:
        print("No versions to analyze.")

    # Remove metadata from all output files except _TN
    output_files = [
        os.path.join(output_folder, f"{base_name}_LB.opus"),
        os.path.join(output_folder, f"{base_name}_CR.flac"),
    ]
    if process_SR:
        output_files.append(os.path.join(output_folder, f"{base_name}_SR_residual.flac"))
    if process_HR:
        output_files.append(os.path.join(output_folder, f"{base_name}_HR_residual.flac"))
    remove_metadata(output_files)

    # File size reporting
    report_file_sizes(input_file_path, output_folder, process_SR)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 FOMA-encoder.py <input_file_path>")
        sys.exit(1)
    input_file_path = sys.argv[1]
    main(input_file_path)
