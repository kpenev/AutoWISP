function showExtracted()
{
    addRegions(
               [
                   {
                       "shape": "circle",
                       "x": 100,
                       "y": 100,
                       "r": 10,
                       "linewidth": 3
                   },
                   {
                       "shape": "rect",
                       "x": 200,
                       "y": 100,
                       "width": 10,
                       "height": 5,
                       "linewidth": 3
                   },
                   {
                       "shape": "x",
                       "x": 300,
                       "y": 100,
                       "width": 10,
                       "height": 5,
                       "linewidth": 3
                   },
                   {
                       "shape": "+",
                       "x": 100,
                       "y": 200,
                       "width": 5,
                       "height": 10,
                       "linewidth": 3
                   },
                   {
                       "shape": "ellipse",
                       "x": 200,
                       "y": 200,
                       "rx": 10,
                       "ry": 5,
                       "linewidth": 3
                   }
               ],
              "px");
}
