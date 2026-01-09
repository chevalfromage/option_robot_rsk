function simulator_initialize(backend, isView) {
    backend.constants(function (constants) {
        let ratio_w = null
        let ratio_h = null
        let back_width = null
        let back_height = null

        function update_ratios() {
            back_width = document.getElementById('back').offsetWidth
            back_height = document.getElementById('back').offsetHeight
            ratio_w = back_width / constants["carpet_length"]
            ratio_h = back_height / constants["carpet_width"]
            ratio_h = Math.min(ratio_w, ratio_h)
        }
        $(window).on("resize", update_ratios)

        function transformViewToSim(position, orientation) {
            let simulatorPos = [0.0, 0.0, 0.0]
            let pos = [position[0], position[1], orientation]
            simulatorPos[0] = ((pos[0]) * ratio_w) + back_width / 2
            simulatorPos[1] = ((-pos[1]) * ratio_h) + back_height / 2
            simulatorPos[2] = round(-pos[2] + Math.PI / 2)
            return simulatorPos
        }

        function isDifferent(lastPos, position) {
            minimumTranslation = 1
            minimumRotation = 0.05
            if (Math.abs(lastPos[0] - position[0]) > minimumTranslation) {
                return true
            } else if (Math.abs(lastPos[1] - position[1]) > minimumTranslation) {
                return true
            } else if (Math.abs(lastPos[2] - position[2]) > minimumRotation) {
                return true
            }
            return false
        }

        function drawLeds(color, context) {
            for (i = -30; i < -30 + 120 * 3; i += 120) {
                angle = i * Math.PI / 180
                x = Math.round(Math.cos(angle) * constants["robot_radius"] * ratio_w * 0.93)
                y = Math.round(Math.sin(angle) * constants["robot_radius"] * ratio_w * 0.93)
                context.beginPath()
                gradient = context.createRadialGradient(x, y, 0, x, y, 70);
                gradient.addColorStop(0.05, "rgba(" + color + ",1)");
                gradient.addColorStop(0.1, "rgba(" + color + ",0.5)");
                gradient.addColorStop(0.25, "rgba(" + color + ",0)");
                context.fillStyle = gradient
                context.fillRect(x - 25, y - 25, 200, 200);
            }
        }

        function drawCircle(position, radius, color, canvas, clear = false, tickness = 0, dash = 0) {
            context = canvas.getContext('2d')
            if (clear) context.clearRect(0, 0, canvas.width, canvas.height)
            context.beginPath()
            context.strokeStyle = color
            context.fillStyle = color
            if (dash != 0) context.setLineDash(dash);
            else context.setLineDash([]);
            context.arc(position[0], position[1], radius, 0, Math.PI * 2);
            context.lineWidth = tickness
            if (tickness == 0) context.fill()
            else context.stroke()
        }

        function drawline(begin, end, canvas, color, tickness = 0) {
            context = canvas.getContext('2d')
            context.beginPath()
            context.strokeStyle = color
            context.fillStyle = color

            context.moveTo(...begin);
            context.lineTo(...end);
            context.lineWidth = tickness
            context.stroke()
        }

        function drawArrow(begin, end, canvas, color) {
            const ctx = canvas.getContext('2d')
            ctx.beginPath()
            ctx.strokeStyle = color
            ctx.lineWidth = 4
            ctx.moveTo(begin[0], begin[1])
            ctx.lineTo(end[0], end[1])
            ctx.stroke()

            const angle = Math.atan2(end[1] - begin[1], end[0] - begin[0])
            const head = 10
            ctx.beginPath()
            ctx.fillStyle = color
            ctx.moveTo(end[0], end[1])
            ctx.lineTo(
                end[0] - head * Math.cos(angle - Math.PI / 6),
                end[1] - head * Math.sin(angle - Math.PI / 6)
            )
            ctx.lineTo(
                end[0] - head * Math.cos(angle + Math.PI / 6),
                end[1] - head * Math.sin(angle + Math.PI / 6)
            )
            ctx.closePath()
            ctx.fill()
        }

        function markerColor(marker) {
            if (marker.startsWith("green")) {
                return "rgba(0, 200, 0, 0.9)"
            } else if (marker.startsWith("blue")) {
                return "rgba(0, 160, 255, 0.9)"
            }
            return "rgba(255, 255, 255, 0.9)"
        }

        function drawBall(position) {
            ball = transformViewToSim(position)
            ballCanvas = document.getElementById("ball")
            ballRadius = constants["ball_radius"] * ratio_w
            drawCircle(ball, ballRadius, "orange", ballCanvas, true)
        }

        function UpdateView() {

            // FPS Limit
            backend.get_state(function (state) {
                if (state.simulated) {
                    tick += 1
                    if (Date.now() - T0 > 100) {
                        $('.fps').text("FPS : " + Math.round(1000 / ((Date.now() - T0) / tick)));
                        T0 = Date.now()
                        tick = 0
                    }
                }

                if (!ratio_w || !ratio_h) {
                    update_ratios()
                }

                let presentMarker = state.markers
                let canvas = document.getElementById("robots")

                if (!("offscreenCanvas" in canvas)) {
                    canvas.offscreenCanvas = document.createElement("canvas")
                }
                canvas.offscreenCanvas.width = canvas.width
                canvas.offscreenCanvas.height = canvas.height

                let context = canvas.offscreenCanvas.getContext("2d")
                context.resetTransform()
                context.clearRect(0, 0, canvas.width, canvas.height)

                // Draw present Robot
                for (var entry in presentMarker) {
                    robot = presentMarker[entry]
                    robotPos = transformViewToSim(robot.position, robot.orientation)

                    // Context placement
                    context.resetTransform()
                    context.translate(robotPos[0], robotPos[1])
                    context.rotate(robotPos[2])

                    // Draw leds
                    if (Object.keys(state["leds"]).length != 0 && state["leds"][entry].length == 3) {
                        markers[entry]["leds"] = state["leds"][entry]
                        for (var i = 0; i < 3; i++) {
                            markers[entry]["leds"][i] = Math.round(Math.min(255, 50 + Math.log(markers[entry]["leds"][i] + 1) / Math.log(256) * 255))
                        }
                        drawLeds(markers[entry]["leds"], context)
                    }

                    let robotSize = constants["robot_radius"] * 2 * ratio_w
                    context.imageSmoothingEnabled = true
                    context.drawImage(markers[entry]["image"], -robotSize / 2, -robotSize / 2, robotSize, robotSize)
                    markers[entry]["pos"] = robotPos
                    markers[entry]["clear"] = false

                    if (
                        state.orders
                        && state.orders[entry]
                        && display_settings["orders_vector"]
                        && display_settings["orders_vector"]["value"]
                    ) {
                        const order = state.orders[entry]
                        if (order.length >= 2) {
                            const orientation = robot.orientation
                            const cos = Math.cos(orientation)
                            const sin = Math.sin(orientation)
                            const worldDx = cos * order[0] - sin * order[1]
                            const worldDy = sin * order[0] + cos * order[1]
                            const magnitude = Math.hypot(worldDx, worldDy)
                            if (magnitude > 1e-3) {
                                const start = [robotPos[0], robotPos[1]]
                                const pxVector = [worldDx * ratio_w, -worldDy * ratio_h]
                                const pixMagnitude = Math.hypot(pxVector[0], pxVector[1]) || 1
                                const length = constants["robot_radius"] * ratio_w * 1.5 * Math.min(1.2, magnitude)
                                const scale = length / pixMagnitude
                                const end = [
                                    start[0] + pxVector[0] * scale,
                                    start[1] + pxVector[1] * scale
                                ]
                                drawArrow(start, end, canvas.offscreenCanvas, "#ff3333")
                            }
                        }
                    }
                }

                canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height)
                canvas.getContext("2d").drawImage(canvas.offscreenCanvas, 0, 0)

                //Draw Ball and placement circle 
                ballCanvas = document.getElementById("ball")
                ballContext = ballCanvas.getContext("2d")

                if (state.ball != null) {
                    drawBall(state.ball)
                }

                if (
                    state.targets
                    && display_settings["goto_targets"]
                    && display_settings["goto_targets"]["value"]
                ) {
                    for (const marker in state.targets) {
                        const target = state.targets[marker]
                        if (!target || !target.position) {
                            continue
                        }
                        const orientation = (typeof target.orientation === "number") ? target.orientation : 0
                        const targetView = transformViewToSim(
                            target.position,
                            orientation
                        )
                        drawCircle(
                            [targetView[0], targetView[1]],
                            constants["robot_radius"] * ratio_w * 0.35,
                            "#ff4d4d",
                            ballCanvas,
                            false,
                            2
                        )
                    }
                }

                let placementCirclePosition = state["referee"]["wait_ball_position"]
                if (placementCirclePosition != null) {
                    drawCircle(transformViewToSim(placementCirclePosition), constants.place_ball_margin * ratio_w, "red", ballCanvas, false, 1)
                }

                if (display_settings["landmark"]["value"]) {
                    center = [ballCanvas.width / 2, ballCanvas.height / 2]
                    drawline(center, [center[0], center[1] - 100], ballCanvas, "green")
                    drawline(center, [center[0] + 100, center[1]], ballCanvas, "red")
                }

                if (display_settings["timed_circle"]["value"]) {
                    drawCircle(transformViewToSim(state.ball), constants.timed_circle_radius * ratio_w, "red", ballCanvas, false, 1, [10, 10])
                }

            });
        }

        function resizeCanvas(canvas) {
            backgroundCanvas = document.getElementById('back')
            canvas.width = backgroundCanvas.offsetWidth
            canvas.height = backgroundCanvas.offsetHeight
            return canvas
        }

        function runView() {
            $('#ViewChange').html("<i class='bi bi-camera'></i> Camera View")
            $('#vision').addClass('d-none')
            $('#back').removeClass('d-none')
            $('.sim_vim').css('opacity', '100')

            // Draw Background
            var background = new Image()
            background.src = "static/imgs/field.svg"
            background.onload = function () {
                let context = document.getElementsByTagName('canvas')[0].getContext('2d')
                context.canvas.width = this.naturalWidth
                context.canvas.height = this.naturalHeight
                context.drawImage(background, 0, 0)
            }

            markers = { "blue1": NaN, "blue2": NaN, "green1": NaN, "green2": NaN }
            for (let marker in markers) {
                markers[marker] = { "image": NaN, "context": NaN, "pos": [0, 0, 0], "leds": [0, 0, 0], "clear": true }
            }
            for (var key in markers) {
                markers[key]["image"] = new Image();
                markers[key]["image"].src = "static/imgs/robot" + key + ".png"
            }
            resizeCanvas(document.getElementById("robots"))
            resizeCanvas(document.getElementById("ball"))

            clearInterval(intervalId)
            intervalId = setInterval(UpdateView, 1000 / fps_limit)
        }

        function clearView() {
            clearInterval(intervalId)
            intervalId = NaN
            $('#ViewChange').html("<i class='bi bi-camera'></i> Simulated View")
            $('#vision').removeClass('d-none')
            $('#back').addClass('d-none')
            $('.sim_vim').css('opacity', '0')
        }

        function switchView() {
            if (isNaN(intervalId)) {
                runView()
            } else {
                clearView()
            }
        }


        function get_display_settings() {
            let html = ''
            for (setting_name in display_settings) {
                let setting = display_settings[setting_name]
                let checked = setting["value"] ? 'checked="checked"' : ''

                html += '<div class="form-check form-switch">'
                html += '    <input class="form-check-input display-setting" type="checkbox"'
                html += 'role="switch" rel="' + setting_name + '" ' + checked + '>'
                html += '    <label class="form-check-label" for="flexSwitchCheckDefault">'
                html += '    ' + setting['label']
                html += '    </label>'
                html += '</div>'
            }
            html += '<div class="range">'
            html += '<label class="form-label" for="flexSwitchCheckDefault">FPS Limit : ' + fps_limit + '</label>'
            html += '<input id="aaaa" type="range" class="form-range" min="10" max="65" step="5" value="' + fps_limit + '", >'
            html += '</div>'

            $('.display-settings').html(html)
            $('.display-setting').click(function () {
                display_settings[$(this).attr('rel')]["value"] = $(this).is(':checked')
            });

            document.querySelector(".range .form-range").addEventListener('input', function (aa) {
                fps_limit = this.value
                clearInterval(intervalId)
                if (fps_limit == 65) {
                    intervalId = setInterval(UpdateView, 1000 / 240)
                    $('.form-label').text("FPS Limit : unlimited")
                } else {
                    intervalId = setInterval(UpdateView, 1000 / fps_limit)
                    $('.form-label').text("FPS Limit : " + fps_limit)
                }
            })

        }
        $('.display-python-settings').click(function () {
            get_display_settings()
        });

        display_settings = {
            "landmark": { "label": "Center Landmark", "default": true, "type": "" },
            "timed_circle": { "label": "Timed Circle", "default": false, "type": "" },
            "goto_targets": { "label": "Goto Targets", "default": true, "type": "" },
            "orders_vector": { "label": "Goto Orders", "default": true, "type": "" },
        }
        for (setting_name in display_settings) {
            display_settings[setting_name]["value"] = display_settings[setting_name]["default"]
        }



        const carpetSize = [constants["carpet_length"], constants["carpet_width"]]
        intervalId = NaN
        let fps_limit = 30
        let selectedObjet = "ball"
        let tick = 0
        let T0 = Date.now()

        if (isView) {
            setTimeout(runView, 1000)
            runView()
        }
        window.onresize = runView

        $('#ViewChange').click(switchView)

        backend.is_simulated(function (isSimulated) {
            if (isSimulated) {
                $('body').addClass('vision-running')
                const canvas = document.getElementById("ball")
                const distance = (x1, y1, x2, y2) => Math.hypot(x2 - x1, y2 - y1);
                let dragType = null
                let initialPosition = null

                function teleportSelectedObjectOnMouse(e) {
                    let reelPos = [0.0, 0.0, 0.0]
                    pos = [...initialPosition]

                    if (dragType == "position") {
                        pos[0] = e.layerX
                        pos[1] = e.layerY
                    } else {
                        pos[2] = Math.atan2(e.layerY - initialPosition[1], e.layerX - initialPosition[0]) + Math.PI / 2
                    }

                    backgroundCanvas = document.getElementById('back')
                    reelPos[0] = (pos[0] - backgroundCanvas.offsetWidth / 2) / ratio_w
                    reelPos[1] = -(pos[1] - backgroundCanvas.offsetHeight / 2) / ratio_h
                    reelPos[2] = -(pos[2] - Math.PI / 2)
                    backend.teleport(selectedObjet, reelPos[0], reelPos[1], reelPos[2])
                }
                canvas.addEventListener("mousedown", function (e) {
                    for (let marker in markers) {
                        if (distance(markers[marker]["pos"][0], markers[marker]["pos"][1], e.layerX, e.layerY) < constants["robot_radius"] * ratio_w) {
                            selectedObjet = marker
                        }
                    }
                    dragType = (e.button == 0) ? "position" : "orientation"
                    if (selectedObjet != "ball") {
                        initialPosition = markers[selectedObjet]["pos"]
                    } else {
                        initialPosition = [0., 0., 0.]
                    }

                    canvas.addEventListener("mousemove", teleportSelectedObjectOnMouse)
                })
                canvas.addEventListener("mouseup", function (e) {
                    teleportSelectedObjectOnMouse(e)
                    canvas.removeEventListener("mousemove", teleportSelectedObjectOnMouse)
                    selectedObjet = "ball"
                })
            }
        })


    })
}