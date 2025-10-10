<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Test Project</title>
    <style>
      #canvas-container {
        display: block;
        width: 100%;
        height: 100%;
      }

      #gravityLabel {
        position: absolute;
        top: 10px;
        left: 10px;
      }

      span {
        color: white;
      }
    </style>
  </head>

  <body>
    <canvas id="canvas-container"></canvas>
    <label class="switch" id="gravityLabel">
      <input type="checkbox" id="gravitySwitch">
      <span class="slider round">Gravity</span>
    </label>
    <script type = module src="triangle.js"></script>
  </body>
</html>
