"""Simple configuration constants for the PhasePaint application."""

# inference parameters
STEPS = 50  # fixed number of diffusion steps
GUIDANCE_SCALE = 7.5  # classifier-free guidance strength
PREVIEW_INTERVAL = 8  # how many steps between each preview
STEP_INTERVAL = 10
START_STEP = 10
GALLERY_SIZE = 720
SHARE = False

# user identifier used for naming the mouse-click log file
USER = "default_user"  # override in your environment or edit directly
