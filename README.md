# Color ASCII Art Converter

A desktop interface for turning images into colored ASCII art, previewing the result, and exporting it as an image or text file.

## Start the App

```bash
pip install -r requirements.txt
python -m ascii_art
```

You can also open an image at launch:

```bash
python -m ascii_art path/to/image.png
```

## Using the Interface

1. Click **Open Image…** and choose an image.
2. Adjust the controls in the right-hand panel. The preview updates automatically.
3. Click **Export Image…** or **Export Text…** to save the result.

The settings panel is scrollable when it is taller than the window.

## Interface Controls

### Character Density

- Set the number of character columns from 20 to 400.
- Use a preset button for a quick density change.
- Higher density preserves more detail.

### Appearance

- **Font size** controls the dimensions of the exported image.
- **Character set** accepts a custom group of characters.
- **Measured Ramp…** displays the current character brightness order.
- **Block Characters** switches to block-style symbols.
- **Full ASCII** uses all printable ASCII characters.
- **Background** selects black, white, or transparent output.

### Brightness and Color

- **Brightness metric** changes how image brightness is interpreted.
- **Candidate count** controls how many denser characters may be considered for each cell.
- **Highlight color preservation** keeps more color in bright areas.
- **Image saturation** ranges from grayscale to more vivid colors.
- **Histogram equalization** can be disabled or applied globally or locally.
- **Local window** and **local clip limit** tune local equalization.
- **Brightness gamma** makes the result brighter or darker.
- **Color mode** selects the character coloring style.
- **Glyph color purity** adjusts glyph color in `pure` mode.
- **Invert** is useful with light backgrounds.

### Preview and Output Information

- The main canvas shows the latest rendered result.
- Transparent output is shown over a checkerboard.
- The information panel reports the character grid, cell size, output size, font, character usage, and render time.

## Export Options

- **Export Image…** saves PNG, WebP, JPEG, or BMP files.
- **Export Text…** saves the ASCII characters as a UTF-8 text file.

Supported input formats include PNG, JPEG, BMP, WebP, GIF, and TIFF.
