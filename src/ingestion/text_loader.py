import os


def read_job_description_from_file(file_path):
    """
    Reads the content of a TXT file and returns it as a string.

    Parameters:
        file_path (str): The path to the TXT file.

    Returns:
        str: The content of the TXT file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            jd_text = file.read()
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
    return jd_text


def extract_all_jobs(root_dir):
    """
    Extracts all job descriptions stored in the specified ROOT_DIR and returns them in a JSON format.

    Parameters:
        root_dir (str): The path to the directory containing raw TXT files of job descriptions.

    Returns:
        list: A list of dictionaries, each representing a job description with keys {"job_id", "file_name", "file_path", "jd_text"}.
    """
    jd_data = []
    for job_id, file_name in enumerate(os.listdir(root_dir)):
        if file_name.endswith(".txt"):
            file_path = os.path.join(root_dir, file_name)
            jd_text = read_job_description_from_file(file_path)
            jd_entry = {
                "job_id": job_id,
                "file_name": file_name,
                "file_path": file_path,
                "jd_text": jd_text,
            }
            jd_data.append(jd_entry)
    return jd_data
