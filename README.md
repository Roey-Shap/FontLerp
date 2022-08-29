# FontLerp

![ezgif com-gif-maker](https://user-images.githubusercontent.com/81587668/187081767-e8549ea3-c2df-4b4c-9eb4-979e2f45f68e.gif)


This is a small project of mine that seeks to answer the question, "how does one linearly interpolate between two arbitrary vector-based fonts"?
 

[Here's the paper on it!](https://pdfhost.io/v/7zebevzBv_FontLerp_Article_1_Export_1)

[Here are some videos of it in action!](https://www.youtube.com/playlist?list=PLRIDiXsuDNc0miu8VrzF59OjqiXGUh6ZN)


Almost all of the code is mine; I used TTFQuery (ripped it apart a bit for my specific needs...) to pull font data from TTF files, and, obviously, didn't create any of the fonts themselves. They're just in there for testing purposes.
You can find the license notice for TTFQuery in "ttfQuery_Notice" in the root directory of this source tree.


This isn't particularly user-friendly, but for those interested, it does work... mostly. The only applicable projection method is what it's currently set to: "Pillow Projection", which is set via string at the start of main.py.

<br>

## Features

Interpolate betweena any two glyphs!
Create text that begins as one font and ends as another!
Hold space to pan/scroll with the mouse to zoom in and out! Experience abstract fonts in their full scalable glory!
Look through vastly improvable code! Wow! Just think of the possibilities!

<br>

In the future, I'd like to expand it to make it easier to use. Most of the features to make it a fully-blown font editor are there, just sorta hidden behind the main font-interpolating showcase (i.e. manipulating points by dragging them around).

<br>

Copyright (c) 2022, Roey Shapiro
All rights reserved.

This source code is licensed under the Apache 2.0-style license found in the LICENSE.md file in the root directory of this source tree.
