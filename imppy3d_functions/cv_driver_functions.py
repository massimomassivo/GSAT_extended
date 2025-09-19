# Import external dependencies
import numpy as np
import cv2 as cv

# Import local packages
import cv_processing_wrappers as wrap
import cv_interactive_processing as ifun


def interact_driver_blur(img_in, fltr_name_in):
    """Preview blur filters interactively for a grayscale image.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` array representing the grayscale source image. The
        input is consumed read-only and is therefore expected to be
        pre-normalised for the intended filter.
    fltr_name_in : str
        Name of the blur filter to preview (``"average"``,
        ``"gaussian"``, ``"median"`` or ``"bilateral"``). The argument is
        case-insensitive.

    Returns
    -------
    numpy.ndarray
        Filtered image that reflects the settings selected when closing
        the GUI window. The array shares the ``uint8`` dtype and shape of
        ``img_in``.

    Side Effects
    ------------
    Opens OpenCV windows with trackbars and prints status messages while
    the session is active.

    Notes
    -----
    More details on the mathematical filters are available in the OpenCV
    tutorial ``https://docs.opencv.org/4.2.0/d4/d13/tutorial_py_filtering.html``.
    Spezifische Sonderfälle (invertierte Grauwerte, Batch-Vorschau) sind
    im Abschnitt ``Sonderfälle und Batch-Hinweise`` der
    ``imppy3d_functions``-README erläutert und werden von den
    ``apply_*``-Funktionen geteilt.
    """

    # ---- Start Local Copies ----
    fltr_name = fltr_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    fltr_list = ["average", "gaussian", "median", "bilateral"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: {fltr_list}\nDefaulting to a Gaussian.")
        fltr_name = "gaussian"

    # -------- Filter Type: "average" --------
    if fltr_name == "average":
        [img_fltr, fltr_params] = ifun.interact_average_filter(img_in)

    # -------- Filter Type: "gaussian" --------
    elif fltr_name == "gaussian":
        [img_fltr, fltr_params] = ifun.interact_gaussian_filter(img_in)

    # -------- Filter Type: "median" --------
    elif fltr_name == "median":
        [img_fltr, fltr_params] = ifun.interact_median_filter(img_in)

    # -------- Filter Type: "bilateral" --------
    elif fltr_name == "bilateral":
        [img_fltr, fltr_params] = ifun.interact_bilateral_filter(img_in)
    
    # Using this function just to write to standard output
    apply_driver_blur(img_in, fltr_params)

    return img_fltr


def apply_driver_blur(img_in, fltr_params_in, quiet_in=False):
    """Apply a configured blur filter without user interaction.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` array containing the grayscale image to be blurred.
        The array is copied internally to preserve the original input.
    fltr_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_blur`. The
        first entry selects the filter (``"average"``, ``"gaussian"``,
        ``"median"`` or ``"bilateral"``); additional entries correspond
        to the filter-specific kernel configuration as described in the
        README section *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        Suppress status messages during batch execution. Defaults to
        ``False``.

    Returns
    -------
    numpy.ndarray
        Blurred image with the same shape and dtype as ``img_in``.

    Side Effects
    ------------
    Writes informational messages to ``stdout`` unless ``quiet_in`` is
    ``True``.

    Notes
    -----
    The OpenCV filter semantics are documented in
    ``https://docs.opencv.org/4.2.0/d4/d13/tutorial_py_filtering.html``.
    Eingabebilder mit invertierten Intensitäten sowie Batch-Workflows
    sind in der ``imppy3d_functions``-README erläutert und gelten auch
    für diese Funktion.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    fltr_name = (fltr_params[0]).lower()

    # Extract the filter parameters depending on the filter type, and
    # then apply said filter. Note, it is possible that some parameters,
    # like kernel size, are zero which would throw an error if used in
    # some of OpenCV's filters. If kernel size is zero, then that means
    # just show the original image -- no changes were made.
    if fltr_name == "average": # Equal-weighted average kernel
        avg_ksize = fltr_params[1]
        if avg_ksize == (0, 0): # No filtering performed, keep original
            return img 
        else: # Apply filter
            img_fltr = cv.blur(img, avg_ksize)

        if not quiet:
             print(f"\nSuccessfully appled the 'average' blur filter:\n"\
                f"    Kernel Size = {(avg_ksize, avg_ksize)}")

    elif fltr_name == "gaussian": # Gaussian-weighted kernel
        gaus_ksize = fltr_params[1]
        gaus_sdev = fltr_params[2]
        if gaus_ksize == (0, 0): # No filtering performed, keep original
            return img 
        else: # Apply filter
            if gaus_sdev < 0: # Calculate automatically if less than 0
                gaus_sdev = 0.3*((gaus_ksize - 1)*0.5 - 1) + 0.8

            if gaus_ksize[0] % 2 == 0: # Must be odd for this filter
                gaus_ksize[0] += -1
                gaus_ksize[1] += -1

            img_fltr = cv.GaussianBlur(img, gaus_ksize, gaus_sdev)

        if not quiet:
            print(f"\nSuccessfully applied the 'gaussian' blur filter:\n"\
                f"    Kernel Size = {(gaus_ksize, gaus_ksize)}\n"\
                f"    Standard Deviation = {gaus_sdev}") 

    elif fltr_name == "median": # Center weight of kernal is median value
        med_ksize = fltr_params[1]
        if med_ksize == 0: # No filtering performed, keep original
            return img 
        else: # Apply filter
            if med_ksize % 2 == 0: # Must be odd for this filter
                med_ksize += -1
            img_fltr = cv.medianBlur(img, med_ksize) 

        if not quiet:
            print(f"\nSuccessfully applied the 'median' blur filter:\n"\
                f"    Kernel Size = {med_ksize}")

    elif fltr_name == "bilateral": # Edge-preserving Gaussian kernel
        bil_dsize = fltr_params[1]
        bil_sint = fltr_params[2]
        if bil_dsize == 0: # No filtering performed, keep original
            return img 
        else: # Apply filter
            if bil_dsize % 2 != 0: # Should be even for this filter
                bil_dsize += 1
            img_fltr = cv.bilateralFilter(img, bil_dsize, bil_sint, bil_sint)

        if not quiet:
            print(f"\nSuccessfully applied the 'bilateral' blur filter:\n"\
                f"    Pixel Neighborhood = {bil_dsize}\n" \
                f"    Intensity Threshold = {bil_sint}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported blur filter.")
        img_fltr = img

    return img_fltr


def interact_driver_sharpen(img_in, fltr_name_in):
    """Preview edge-enhancing filters interactively.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image that serves as the preview source.
        The data is consumed read-only by the interactive widgets.
    fltr_name_in : str
        Identifier of the sharpening filter (``"unsharp"``, ``"laplacian"``
        or ``"canny"``). Comparison between the modes is case-insensitive.

    Returns
    -------
    numpy.ndarray
        Sharpened image computed when the OpenCV window is closed via the
        Enter or Escape key.

    Side Effects
    ------------
    Opens an OpenCV preview window with trackbars and writes status
    information to ``stdout``.

    Notes
    -----
    Die Parameter, die die Trackbars liefern, entsprechen exakt den
    Eingaben für :func:`apply_driver_sharpen`. Ergänzende Hinweise zu
    invertierten Grauwerten und Batch-Abläufen finden sich im Abschnitt
    ``Sonderfälle und Batch-Hinweise`` der
    ``imppy3d_functions``-README.
    """

    # ---- Start Local Copies ----
    fltr_name = fltr_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    fltr_list = ["unsharp", "laplacian", "canny"]

    # Ensure that the filter name is recognized
    if fltr_name not in fltr_list:
        print(f"\n{fltr_name_in} is not a supported filter type. Supported "\
            f"filter types are: {fltr_list}\nDefaulting to an Unsharp Mask.")
        fltr_name = "unsharp"

    # -------- Filter Type: "unsharp" --------
    if fltr_name == "unsharp":
        [img_sharp, sharp_params] = ifun.interact_unsharp_mask(img_in)

    # -------- Filter Type: "laplacian" --------
    elif fltr_name == "laplacian":
        [img_sharp, sharp_params] = ifun.interact_laplacian_sharp(img_in)

    # -------- Filter Type: "canny" --------
    elif fltr_name == "canny":
        [img_sharp, sharp_params] = ifun.interact_canny_sharp(img_in)

    # Using this function just to write to standard output
    apply_driver_sharpen(img_in, sharp_params)

    return img_sharp


def apply_driver_sharpen(img_in, fltr_params_in, quiet_in=False):
    """Apply edge-enhancing filters with predefined parameters.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image that will be sharpened. A copy is
        created to prevent modifications to the original array.
    fltr_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_sharpen`.
        Depending on the first entry (``"unsharp"``, ``"laplacian"`` or
        ``"canny"``) the remaining parameters encode kernel radii,
        blending amounts and thresholds. Eine tabellarische Übersicht
        findet sich in der README unter *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        Suppress status messages that describe the applied settings.
        Defaults to ``False``.

    Returns
    -------
    numpy.ndarray
        Sharpened image with the same shape and dtype as ``img_in``.

    Side Effects
    ------------
    Emits status messages via ``stdout`` unless ``quiet_in`` is ``True``.

    Notes
    -----
    Sharpening entspricht einer gezielten Kantengewichtung. Hinweise zu
    invertierten Grauwerten und den Auswirkungen auf Batch-Läufe sind im
    Abschnitt ``Sonderfälle und Batch-Hinweise`` der
    ``imppy3d_functions``-README gesammelt.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    fltr_params = fltr_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    fltr_name = (fltr_params[0]).lower()

    # -------- Filter Type: "unsharp" --------
    if fltr_name == "unsharp":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[1] % 2 == 0: # Must be odd for the blur filters
            fltr_params_2[1] += -1

        if fltr_params_2[0] <= 0:
            return img
        elif fltr_params_2[1] < 1:
            return img

        img_sharp = wrap.unsharp_mask(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'unsharp' mask:\n"\
                f"    Amount: {fltr_params_2[0]}%\n"\
                f"    Radius of Blur Kernel: {fltr_params_2[1]}")

            if fltr_params_2[2]:
                print("    Blur Filter Type: Median")
            else:
                print("    Blur Filter Type: Gaussian\n"\
                    f"    Standard Deviation: {fltr_params_2[3]}")

    # -------- Filter Type: "laplacian" --------
    elif fltr_name == "laplacian":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[1] % 2 == 0: # Must be odd for the Gaussian blur
            fltr_params_2[1] += -1

        if fltr_params_2[2] % 2 == 0: 
            fltr_params_2[2] += -1

        if fltr_params_2[0] <= 0:
            return img
        elif (fltr_params_2[2] < 1) or (fltr_params_2[1] < 1):
            return img

        img_sharp = wrap.laplacian_sharp(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'laplacian' mask:\n"\
                f"    Amount: {fltr_params_2[0]}%\n"\
                f"    Radius of Blur Kernel: {fltr_params_2[1]}\n"\
                f"    Radius of Laplacian Kernel: {fltr_params_2[2]}")

            if fltr_params_2[3]:
                print("    Blur Filter Type: Median")
            else:
                print("    Blur Filter Type: Gaussian\n"\
                    f"    Standard Deviation: {fltr_params_2[4]}")

    # -------- Filter Type: "canny" --------
    elif fltr_name == "canny":
        fltr_params_2 = fltr_params[1:]

        if fltr_params_2[0] % 2 == 0: 
            fltr_params_2[0] += -1

        img_sharp = wrap.canny_sharp(img, fltr_params_2)

        if not quiet:
            print(f"\nSuccessfully applied the 'canny' edge mask:\n"\
                f"    Radius of Canny Edge Kernel: {fltr_params_2[0]}\n"\
                f"    Amount: {fltr_params_2[1]}%\n"\
                f"    Hysteresis Threshold 1: {fltr_params_2[2]}\n"\
                f"    Hysteresis Threshold 2: {fltr_params_2[3]}")

    else:
        print(f"\nERROR: {fltr_name} is not a supported sharpen filter.")
        img_sharp = img
        
    return img_sharp


def interact_driver_equalize(img_in, eq_name_in):
    """Preview histogram equalisation workflows interactively.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image used as the live preview input.
    eq_name_in : str
        Equalisation strategy to evaluate (``"global"`` or ``"adaptive"``).
        The value is interpreted case-insensitively.

    Returns
    -------
    numpy.ndarray
        Equalised image corresponding to the parameters chosen in the
        interactive session.

    Side Effects
    ------------
    Opens OpenCV preview windows and writes user guidance to ``stdout``.

    Notes
    -----
    ``"global"`` equalisation wendet unmittelbar den OpenCV-Standard an
    und besitzt daher keine Trackbars. Für Hinweise zu invertierten
    Grauwerten und Batch-Kontexten siehe Abschnitt ``Sonderfälle und
    Batch-Hinweise`` der ``imppy3d_functions``-README. The underlying
    algorithms are described in
    ``https://docs.opencv.org/4.2.0/d5/daf/tutorial_py_histogram_equalization.html``.
    """

    # ---- Start Local Copies ----
    eq_name = eq_name_in.lower() # Ensure string is all lowercase
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    eq_list = ["global", "adaptive"]

    # Ensure that the filter name is recognized
    if eq_name not in eq_list:
        print(f"\n{eq_name_in} is not a supported filter type. Supported "\
            f"filter types are: {eq_list}\nDefaulting to a Adaptive.")
        eq_name = "adaptive"

    # -------- Filter Type: "global" --------
    if eq_name == "global":
        img_eq = wrap.global_equalize(img_in)
        eq_params = ["global"]

    # -------- Filter Type: "adaptive" --------
    elif eq_name == "adaptive":
        [img_eq, eq_params] = ifun.interact_adaptive_equalize(img_in)

    # Using this function just to write to standard output
    apply_driver_equalize(img_in, eq_params)

    return img_eq


def apply_driver_equalize(img_in, eq_params_in, quiet_in=False):
    """Equalise image intensities using preset parameters.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image to be contrast-enhanced. A copy is
        taken before processing.
    eq_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_equalize`. The
        first entry specifies the mode (``"global"`` or ``"adaptive"``),
        followed by optional arguments such as ``clip_limit`` and
        ``grid_size``. A structured summary is provided in der README im
        Abschnitt *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        When ``True``, suppress informational print statements. Defaults
        to ``False``.

    Returns
    -------
    numpy.ndarray
        Equalised image with the same shape and dtype as ``img_in``.

    Side Effects
    ------------
    Emits status messages via ``stdout`` unless ``quiet_in`` disables
    them.

    Notes
    -----
    Equalisation algorithms follow the OpenCV implementations described
    in ``https://docs.opencv.org/4.2.0/d5/daf/tutorial_py_histogram_equalization.html``.
    Hinweise zu invertierten Segmentierungen und Batch-Pipelines sind im
    README-Abschnitt ``Sonderfälle und Batch-Hinweise`` gebündelt.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    eq_params = eq_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    eq_name = (eq_params[0]).lower()

    # -------- Filter Type: "global" --------
    if eq_name == "global":
        img_eq = wrap.global_equalize(img)
        if not quiet:
            print("\nSuccessfully applied 'global' equalization")

    # -------- Filter Type: "adaptive" --------
    elif eq_name == "adaptive":
        eq_params_2 = eq_params[1:]

        if eq_params_2[1] < 1:
            return img

        img_eq = wrap.adaptive_equalize(img, eq_params_2)

        if not quiet:
            print("\nSuccessfully applied locally 'adaptive' equalization:\n"\
                f"    Clip Limit: {eq_params_2[0]}\n"\
                f"    Tile Size: ({eq_params_2[1]},{eq_params_2[1]})")

    else:
        print(f"\nERROR: {eq_name} is not a supported equalization "\
            "operation.")
        img_eq = img

    return img_eq


def interact_driver_morph(img_in):
    """Interactively explore morphological operations.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image that serves as the baseline for the
        preview session.

    Returns
    -------
    numpy.ndarray
        Morphologically transformed image reflecting the state at the
        time the preview window is closed.

    Side Effects
    ------------
    Opens OpenCV windows with trackbars and prints textual hints to the
    console.

    Notes
    -----
    Die verfügbaren Operationen (Erosion, Dilatation, Opening, Closing)
    sind identisch zu :func:`apply_driver_morph`. Hinweise zu Batch- und
    Invertierungs-Szenarien sind im Abschnitt ``Sonderfälle und
    Batch-Hinweise`` der README aufgeführt.
    """

    [img_morph, morph_params] = ifun.interact_morph(img_in)

    # Using this function just to write to standard output
    apply_driver_morph(img_in, morph_params)

    return img_morph


def apply_driver_morph(img_in, morph_params_in, quiet_in=False):
    """Execute morphological filters with predefined settings.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` image expected to contain a binary mask.
    morph_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_morph`. The
        first entry selects the operation (0 = dilation, 1 = erosion,
        2 = opening, 3 = closing), followed by structuring-element
        metadata such as ``se_shape``, ``kernel_radius`` and optional
        iteration counts. Eine Kurzreferenz befindet sich in der README
        unter *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        When ``True`` suppress informational logging. Defaults to
        ``False``.

    Returns
    -------
    numpy.ndarray
        Morphologically processed image with the same shape and dtype as
        ``img_in``.

    Side Effects
    ------------
    Prints status information to ``stdout`` unless ``quiet_in`` disables
    the output.

    Notes
    -----
    Morphologische Operatoren erwarten binäre Masken (0/255). Hinweise
    auf Sonderfälle wie invertierte Masken oder aufeinanderfolgende
    Batch-Läufe entnehmen Sie dem Abschnitt ``Sonderfälle und
    Batch-Hinweise`` der README.
    """

    # ---- Start Local Copies ----
    morph_params = morph_params_in
    quiet = quiet_in

    # 0 is open => erosion followed by dilation (removes noise)
    # 1 is close => dilation followed by erosion (closes small holes)
    flag_open_close = morph_params[0]

    # 0 is rectangular kernel, 1 is elliptical kernel
    flag_rect_ellps = morph_params[1]

    # Width (or diameter) of the kernel in pixels
    k_size = morph_params[2]

    # Number of iterations of the erode and dilation operations
    num_erode = morph_params[3]
    num_dilate = morph_params[4]

    if k_size < 1: # Nothing to report if kept original image
        return img_in
    elif (num_dilate < 1) and (num_erode < 1):
        return img_in

    # Perform successive morphological operations
    img_morph = wrap.multi_morph(img_in, morph_params)

    # Write out comments to standard output based on what was done
    if flag_open_close: # Close
        if (num_dilate >= 1) and (num_erode >= 1):
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_dilate} X Dilate   followed by   "\
                    f"{num_erode} X Erode"
                print("    Kernel Operations: " + op_hist)

        elif num_dilate < 1: # Only do a erosion
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"0 X Dilate   followed by   "\
                    f"{num_erode} X Erode"
                print("    Kernel Operations: " + op_hist)

        else: # Only do an dilation
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_dilate} X Dilate   followed by   "\
                    f"0 X Erode"
                print("    Kernel Operations: " + op_hist)

    else: # Open
        if (num_dilate >= 1) and (num_erode >= 1):
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_erode} X Erode   followed by   "\
                    f"{num_dilate} X Dilate"
                print("    Kernel Operations: " + op_hist)

        elif num_dilate < 1: # Only do a erosion
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"{num_erode} X Erode   followed by   "\
                    f"0 X Dilate"
                print("    Kernel Operations: " + op_hist)

        else: # Only do an dilation
            if not quiet:
                print("\nSuccessfully applied morphological operations:")
                if flag_rect_ellps:
                    print("    Kernel Shape: Ellipse\n"\
                         f"    Kernel Diameter: {k_size}")
                else:
                    print("    Kernel Shape: Rectangle\n"\
                         f"    Kernel Width: {k_size}")

                op_hist = f"0 X Erode   followed by   "\
                    f"{num_dilate} X Dilate"
                print("    Kernel Operations: " + op_hist)

    return img_morph


def interact_driver_thresh(img_in, thsh_name_in):
    """Interactively derive binarisation parameters.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image to threshold.
    thsh_name_in : str
        Thresholding strategy (``"global"``, ``"adaptive_mean"`` or
        ``"adaptive_gaussian"``). Comparison is case-insensitive.

    Returns
    -------
    numpy.ndarray
        Binary image produced when the interactive session is closed.

    Side Effects
    ------------
    Opens OpenCV GUI elements and prints guidance to ``stdout``.

    Notes
    -----
    ``"global"`` thresholding unterstützt sowohl manuelle Werte als auch
    Otsu-Autodetektion. Hinweise zu invertierten Grauwerteingaben und
    Batch-Verarbeitung finden sich im Abschnitt ``Sonderfälle und
    Batch-Hinweise`` der README. The underlying algorithms are outlined
    in ``https://docs.opencv.org/4.2.0/d7/d4d/tutorial_py_thresholding.html``.
    """

    # ---- Start Local Copies ----
    thsh_name = thsh_name_in.lower()
    # ---- End Local Copies ----

    # List of strings of the supported filter types
    thsh_list = ["global", "adaptive_mean", "adaptive_gaussian"]

    # Ensure that the filter name is recognized
    if thsh_name not in thsh_list:
        print(f"\n{thsh_name} is not a supported threshold type. Supported "\
            f"threshold types are: {thsh_list}\nDefaulting to "\
            "adaptive_gaussian.")
        thsh_name = "adaptive_gaussian"

    if thsh_name == "global":
        [img_thsh, thsh_params] = ifun.interact_global_threshold(img_in)

    elif thsh_name == "adaptive_mean":
        [img_thsh, thsh_params] = ifun.interact_mean_threshold(img_in)

    elif thsh_name == "adaptive_gaussian":
        [img_thsh, thsh_params] = ifun.interact_gaussian_threshold(img_in)

    # Using this function just to write to standard output
    apply_driver_thresh(img_in, thsh_params)

    return img_thsh


def apply_driver_thresh(img_in, thsh_params_in, quiet_in=False):
    """Apply binary thresholding using saved parameters.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image that will be binarised.
    thsh_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_thresh`. The
        first entry selects the strategy (``"global"``, ``"adaptive_mean"``
        or ``"adaptive_gaussian"``), followed by the required thresholds
        or window sizes. Eine Übersichtstabelle steht im README-Abschnitt
        *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        Suppress informational output when ``True``. Defaults to
        ``False``.

    Returns
    -------
    numpy.ndarray
        Binary mask with values ``0`` and ``255`` matching the shape of
        ``img_in``.

    Side Effects
    ------------
    Writes status information to ``stdout`` unless ``quiet_in`` disables
    it.

    Notes
    -----
    Invertierte Eingabebilder sowie Batch-Verarbeitungen werden im
    Abschnitt ``Sonderfälle und Batch-Hinweise`` der README erläutert.
    Globale Thresholds können mit Otsu automatisch bestimmt werden; die
    adaptive Variante erwartet ungerade Fenstergrößen.
    """

    # ---- Start Local Copies ----
    img = img_in.copy()
    thsh_params = thsh_params_in
    quiet = quiet_in
    # ---- End Local Copies ----

    thsh_name = (thsh_params[0]).lower()

    # -------- Filter Type: "global" --------
    if thsh_name == "global":
        thsh_params_2 = thsh_params[1:]

        # If the threshold value is < 0, apply automatic Otsu method
        if thsh_params_2[0] < 0:
            [ret_thsh, img_thsh] = cv.threshold(img, 0, 255, 
                cv.THRESH_BINARY+cv.THRESH_OTSU)

            if not quiet:
                print(f"\nSuccessfully applied 'global' thresholding:\n"\
                    f"    Threshold Value (Otsu Method): {ret_thsh}") 

        else:
            if thsh_params_2[0] > 255:
                thsh_params_2[0] = 255

            [ret_thsh, img_thsh] = cv.threshold(img, thsh_params_2[0], 255, 
                cv.THRESH_BINARY)

            if not quiet:
                print(f"\nSuccessfully applied 'global' thresholding:\n"\
                    f"    Threshold Value: {thsh_params_2[0]}")         

        return img_thsh


    # -------- Filter Type: "adaptive_mean" --------
    elif thsh_name == "adaptive_mean":
        thsh_params_2 = thsh_params[1:]

        if thsh_params_2[0] <= 0: # Return original image
            img_thsh = img

        else:
            if thsh_params_2[0] % 2 == 0: # Must be odd (but not 1) and > 0
                thsh_params_2[0] += -1

            if thsh_params_2[0] < 3:
                thsh_params_2[0] = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 
                thsh_params_2[0], thsh_params_2[1])

            if not quiet:
                print(f"\nSuccessfully applied 'adaptive mean' threshold:\n"\
                    f"    Block Size: {thsh_params_2[0]} pixels\n"\
                    f"    Intensity Offset: {thsh_params_2[1]}")

        return img_thsh

    # -------- Filter Type: "adaptive_gaussian" --------
    elif thsh_name == "adaptive_gaussian":
        thsh_params_2 = thsh_params[1:]

        if thsh_params_2[0] <= 0: # Return original image
            img_thsh = img

        else:
            if thsh_params_2[0] % 2 == 0: # Must be odd (but not 1) and > 0
                thsh_params_2[0] += -1

            if thsh_params_2[0] < 3:
                thsh_params_2[0] = 3

            img_thsh = cv.adaptiveThreshold(img, 255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 
                thsh_params_2[0], thsh_params_2[1])

            if not quiet:
                print(f"\nSuccessfully applied 'adaptive gaussian' "\
                    f"threshold:\n"\
                    f"    Block Size: {thsh_params_2[0]} pixels\n"\
                    f"    Intensity Offset: {thsh_params_2[1]}")

        return img_thsh

    else:
        print(f"\nERROR: {thsh_name} is not a supported threshold "\
            "operation.")


def interact_driver_blob_fill(img_in):
    """Interactively tune blob removal on binary masks.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` image expected to contain values ``0`` and ``255``.

    Returns
    -------
    numpy.ndarray
        Image with undesired blobs removed according to the selected
        thresholds.

    Side Effects
    ------------
    Opens OpenCV preview windows and prints status information.

    Notes
    -----
    Die Parameter (Fläche, Rundheit, Aspektverhältnis) entsprechen den
    Eingaben für :func:`apply_driver_blob_fill`. Hinweise zu invertierten
    Masken und Batch-Verarbeitung finden sich im README-Abschnitt
    ``Sonderfälle und Batch-Hinweise``.
    """
    [img_blob, blob_params] = ifun.interact_blob_fill(img_in)

    # Using this function just to write to standard output
    apply_driver_blob_fill(img_in, blob_params)

    return img_blob


def apply_driver_blob_fill(img_in, blob_params_in, quiet_in=False):
    """Remove labelled blobs based on geometric thresholds.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` binary image to clean.
    blob_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_blob_fill`.
        Encodes area bounds, circularity range, aspect-ratio range and
        replacement colour ``(B, G, R)``. Details are tabulated in der
        README unter *Sonderfälle und Batch-Hinweise*.
    quiet_in : bool, optional
        Suppress console output when ``True``. Defaults to ``False``.

    Returns
    -------
    numpy.ndarray
        Cleaned binary image with the same shape and dtype as ``img_in``.

    Side Effects
    ------------
    Prints progress information to ``stdout`` unless ``quiet_in`` is set.

    Notes
    -----
    Die Funktion ersetzt erkannte Blobs durch den angegebenen Farbwert
    und belässt die übrigen Pixel unverändert. Hinweise zu invertierten
    Masken und Batch-Verarbeitung finden sich im README-Abschnitt
    ``Sonderfälle und Batch-Hinweise``.
    """
    
    img = img_in.copy()
    blob_params = blob_params_in
    quiet = quiet_in

    area_thresh_min = blob_params[0]
    area_thresh_max = blob_params[1]
    circty_thresh_min = blob_params[2]
    circty_thresh_max = blob_params[3]
    ar_min = blob_params[4]
    ar_max = blob_params[5]
    blob_color = blob_params[6]

    PI = float(3.14159265358979323846264338327950288419716939937510)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    area_thresh_min = int(round(area_thresh_min))
    area_thresh_max = int(round(area_thresh_max))

    if area_thresh_min > area_thresh_max:
        area_thresh_min = area_thresh_max
    elif area_thresh_max < area_thresh_min:
        area_thresh_max = area_thresh_min

    if circty_thresh_min > circty_thresh_max:
        circty_thresh_min = circty_thresh_max
    elif circty_thresh_max < circty_thresh_min:
        circty_thresh_max = circty_thresh_min

    if ar_min > ar_max:
        ar_min = ar_max
    elif ar_max < ar_min:
        ar_max = ar_min

    [img_contrs, contr_hierarchy] = cv.findContours(img, cv.RETR_LIST, 
        cv.CHAIN_APPROX_SIMPLE)

    if img_contrs:

        contrs_area = []
        contrs_perim = []
        contrs_circty = []
        contrs_del = []

        for contr in img_contrs:
            cur_area = cv.contourArea(contr)
            contrs_area.append(cur_area)

            cur_perim = cv.arcLength(contr, True)
            contrs_perim.append(cur_perim)

            if cur_perim != 0:
                cur_circty = 4.0*PI*cur_area/(cur_perim*cur_perim)
            else:
                cur_circty = 0.0

            # Could also get the aspect ratio and compare against that
            [x_rect, y_rect, w_rect, h_rect] = cv.boundingRect(contr)
            cur_ar = float(w_rect)/h_rect 

            if (area_thresh_min <= cur_area <= area_thresh_max) and\
                (circty_thresh_min <= cur_circty <= circty_thresh_max) and\
                (ar_min <= cur_ar <= ar_max):
             
                contrs_del.append(contr)        

    # Fill the white blobs with black pixels, effectively deleting them
    if contrs_del: 
        for contr in contrs_del:
            cv.drawContours(img, [contr], 0, blob_color, thickness=cv.FILLED, 
                lineType=cv.LINE_8)

    num_del_blob = len(contrs_del)

    if not quiet:
        print(f"\nSuccessfully filled in {num_del_blob} blobs using the "\
            "following criterion:\n"\
            f"    Min. Area Threshold: {area_thresh_min} pixels\n"\
            f"    Max. Area Threshold: {area_thresh_max} pixels\n"\
            f"    Min. Circularity: {circty_thresh_min}\n"\
            f"    Max. Circularity: {circty_thresh_max}\n"\
            f"    Min. Aspect Ratio: {ar_min}\n"\
            f"    Max. Aspect Ratio: {ar_max}")

    return img


def interact_driver_denoise(img_in):
    """Interactively configure non-local means denoising.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image used as preview input.

    Returns
    -------
    numpy.ndarray
        Denoised image corresponding to the last slider configuration.

    Side Effects
    ------------
    Opens OpenCV preview windows and prints runtime hints. Large window
    sizes may momentarily block UI updates due to the expensive
    computation.

    Notes
    -----
    Die Trackbars steuern Filterstärke, Patch- und Suchfenstergröße und
    entsprechen den Parametern von :func:`apply_driver_denoise`.
    Sonderfälle (invertierte Grauwerte, Batch-Workflows) sind in der
    README unter ``Sonderfälle und Batch-Hinweise`` beschrieben.
    """

    [img_denoise, denoise_params] = ifun.interact_denoise(img_in)

    # Using this function just to write to standard output
    apply_driver_denoise(img_in, denoise_params)

    return img_denoise


def apply_driver_denoise(img_in, denoise_params_in, quiet_in=False):
    """Apply non-local means denoising with fixed parameters.

    Parameters
    ----------
    img_in : numpy.ndarray
        2-D ``uint8`` grayscale image to denoise.
    denoise_params_in : Sequence
        Parameter tuple emitted by :func:`interact_driver_denoise`.
        Contains filter strength ``h``, patch size and search window size.
        Weitere Details stehen im README-Abschnitt *Sonderfälle und
        Batch-Hinweise*.
    quiet_in : bool, optional
        Suppress informational output when ``True``. Defaults to
        ``False``.

    Returns
    -------
    numpy.ndarray
        Denoised image with the same shape and dtype as ``img_in``.

    Side Effects
    ------------
    Prints status messages to ``stdout`` unless ``quiet_in`` is set.

    Notes
    -----
    Größere Suchfenster erhöhen die Laufzeit signifikant. Hinweise zu
    invertierten Grauwerteingaben und Batch-Pipelines entnehmen Sie dem
    README-Abschnitt ``Sonderfälle und Batch-Hinweise``.
    """

    img = img_in.copy()
    denoise_params = denoise_params_in
    quiet = quiet_in

    cur_h = denoise_params[0]
    cur_tsize = denoise_params[1]
    cur_wsize = denoise_params[2]

    if cur_h <= 0:
        img_denoise = img

    else:
        # Must be odd for the denoise function
        if cur_tsize % 2 == 0: 
            cur_tsize += -1

        if cur_wsize % 2 == 0: 
            cur_wsize += -1

        img_denoise = cv.fastNlMeansDenoising(img, h=cur_h, 
            templateWindowSize=cur_tsize, searchWindowSize=cur_wsize)

        if not quiet:
            print("\nSuccessfully applied the non-local means denoising"\
                "filter:\n"\
                f"    Filter Strength: {cur_h}\n"\
                f"    Template Patch Size: {cur_tsize}\n"
                f"    Current Window Size: {cur_wsize}")

    return img_denoise
