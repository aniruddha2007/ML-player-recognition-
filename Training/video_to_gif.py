from moviepy.editor import VideoFileClip
clip = askopenfilename()
clip.write_gif("output.gif")