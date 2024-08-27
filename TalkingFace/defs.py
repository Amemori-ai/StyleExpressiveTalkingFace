from .aligner import aligner
def kernel(
            rank,
            world_size,
            config_path,
            save_path,
            resume_path
          ):
    return aligner(
                   config_path, 
                   save_path, 
                   resume_path, 
                   rank = rank, 
                   world_size = world_size
                  )
