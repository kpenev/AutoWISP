(function () {
    "use strict";

    const TUTORIAL_VERSION = 2;
    const STORAGE_STATE_KEY = "autowisp_bui_tutorial_state";
    const STORAGE_PROMPTED_KEY = "autowisp_bui_tutorial_prompted";
    const STORAGE_CARD_POSITION_KEY = "autowisp_bui_tutorial_card_position";

    const TRANSIENT_PATHS = [
        /^\/change_master_config\/[^/]+\/?$/,
        /^\/configuration\/(?:update_survey_component|update_survey_component_type|delete_from_survey|change_access)\/.*$/,
    ];

    const STEPS = {
        home_new_project: {
            path: /^\/$/,
            goTo: "/",
            selector: "#tutorial-new-project",
            actionSelector: "#tutorial-new-project",
            title: "Create A New Project",
            body: "Click New Project to start a new reduction workflow.",
        },
        create_project_details: {
            path: /^\/new_project\/?$/,
            goTo: "/new_project",
            selectors: ["#project-name", "#project-description"],
            title: "Project Name And Description",
            body: "Enter both project name and description here.",
        },
        create_project_home: {
            path: /^\/new_project\/?$/,
            goTo: "/new_project",
            selector: "#tutorial-project-home",
            actionSelector: "#tutorial-project-home",
            title: "Choose Project Home",
            body: "Click Project Home and pick the folder where outputs will be stored.",
        },
        select_project_home: {
            path: /^\/select_project_home(?:\/.*)?$/,
            selector: "#tutorial-set-project-home",
            actionSelector: "#tutorial-set-project-home",
            title: "Set Project Home",
            body: "Select a directory and click Set Project Home.",
        },
        sony_custom_config: {
            path: /^\/new_project\/?$/,
            goTo: "/new_project",
            selectors: ["#tutorial-custom-config-file", "#custom-config"],
            title: "Sony: Load Custom Config",
            body: "For Sony, load or paste your custom configuration in this optional section before creating the project.",
        },
        create_project_calibration: {
            path: /^\/new_project\/?$/,
            goTo: "/new_project",
            selector: "#tutorial-master-config",
            title: "Calibration Availability",
            body: "Enable or disable bias, dark, and flat based on what calibration frames you have.",
        },
        create_project_submit: {
            path: /^\/new_project\/?$/,
            goTo: "/new_project",
            selector: "#create-project-submit",
            actionSelector: "#create-project-submit",
            title: "Create Project",
            body: "Create the project after finishing setup above.",
        },
        home_open_project: {
            path: /^\/$/,
            goTo: "/",
            resolveTargets: resolveNewestProjectRow,
            actionUsesTargets: true,
            title: "Open The Newest Project",
            body: "Open the newest project in the list to continue.",
        },
        processing_edit_survey: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-edit-survey",
            actionSelector: "#tutorial-edit-survey",
            title: "Edit Survey",
            body: "First after project creation: edit survey.",
        },
        survey_overview: {
            path: /^\/configuration\/survey(?:\/.*)?$/,
            selector: "#tutorial-survey-panel",
            title: "Survey Setup",
            body: "Fill or import survey info, then return to Processing Status.",
        },
        survey_back_processing: {
            path: /^\/configuration\/survey(?:\/.*)?$/,
            selector: "#tutorial-processing-status",
            actionSelector: "#tutorial-processing-status",
            title: "Back To Processing",
            body: "Go back to Processing Status.",
        },
        processing_configure: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-configure-processing",
            actionSelector: "#tutorial-configure-processing",
            title: "PANOPTES: Open Configuration",
            body: "For PANOPTES, go to Configure Processing before adding images.",
        },
        config_import: {
            path: /^\/configuration\/(?!survey)(?:.*)?$/,
            selector: "#tutorial-config-import",
            title: "PANOPTES: Import Config",
            body: "Import your PANOPTES configuration file.",
        },
        config_api_key: {
            path: /^\/configuration\/(?!survey)(?:.*)?$/,
            selector: "#chart-container",
            title: "PANOPTES: Set API Key",
            body: "Update the astrometry API key in configuration before processing.",
        },
        config_back_processing: {
            path: /^\/configuration\/(?!survey)(?:.*)?$/,
            selector: "#tutorial-config-processing-status",
            actionSelector: "#tutorial-config-processing-status",
            title: "Back To Processing",
            body: "Return to Processing Status after import and API key updates.",
        },
        processing_add_images: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-add-images",
            actionSelector: "#tutorial-add-images",
            title: "Add Raw Images",
            body: "Now add raw images. Do this only after the required setup steps.",
        },
        raw_select_files: {
            path: /^\/processing\/select_raw_images(?:\/.*)?$/,
            selector: "#tutorial-file-selector",
            title: "Select Files",
            body: "Select raw FITS files or folders.",
        },
        raw_add_selected: {
            path: /^\/processing\/select_raw_images(?:\/.*)?$/,
            selector: "#tutorial-add-selected-images",
            actionSelector: "#tutorial-add-selected-images",
            title: "Import Selected Images",
            body: "Add the selected files to the project.",
        },
        processing_start: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-start-processing",
            actionSelector: "#tutorial-start-processing",
            title: "Start Processing",
            body: "Start the pipeline.",
        },
        processing_logs: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: ".tutorial-first-log",
            waitForTarget: true,
            title: "Monitor Logs",
            body: "Use logs to monitor status and diagnose issues.",
        },
        processing_object: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-object-button",
            actionSelector: "#tutorial-object-button",
            waitForTarget: true,
            title: "Select Photometric Reference",
            body: "When OBJECT appears, open it to pick reference images.",
        },
        photref_target: {
            path: /^\/processing\/select_photref_target\/?$/,
            selector: ".tutorial-photref-target-row",
            actionSelector: ".tutorial-photref-target-row",
            title: "Choose Target",
            body: "Choose the target row to inspect candidate reference images.",
        },
        photref_image: {
            path: /^\/processing\/select_photref_image(?:\/.*)?$/,
            selector: "#tutorial-set-photref-image",
            actionSelector: "#tutorial-set-photref-image",
            title: "Set Reference Image",
            body: "Set this image as reference and return to Processing Status.",
        },
        processing_resume: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-start-processing",
            actionSelector: "#tutorial-start-processing",
            title: "Resume Processing",
            body: "Start processing again to continue after reference selection.",
        },
        processing_review_results: {
            path: /^\/processing(?:\/\d+)?\/?$/,
            selector: "#tutorial-review-results",
            actionSelector: "#tutorial-review-results",
            title: "Open Results",
            body: "Open Review Results to plot your first light curve.",
        },
        results_enter_id: {
            path: /^\/results\/?$/,
            selectors: ["#star-id-type", "#star-id"],
            title: "Select A Target",
            body: "Choose ID type and enter a target identifier.",
        },
        results_apply: {
            path: /^\/results\/?$/,
            selector: "#apply",
            actionSelector: "#apply",
            title: "Plot First Light Curve",
            body: "Click Apply to render the first light curve.",
        },
        complete: {
            path: /.*/,
            title: "Tutorial Complete",
            body: "Tutorial finished. Use Tutorial on the project list page anytime to restart.",
            isTerminal: true,
        },
    };

    let ui = null;
    let highlightedElements = [];
    let waitTimerId = null;
    let actionCleanup = [];
    let manualCardPosition = null;

    function getFlow(track) {
        const flow = [
            "home_new_project",
            "create_project_details",
            "create_project_home",
            "select_project_home",
        ];

        if (track === "sony") {
            flow.push("sony_custom_config");
        }

        flow.push(
            "create_project_calibration",
            "create_project_submit",
            "home_open_project",
            "processing_edit_survey",
            "survey_overview",
            "survey_back_processing"
        );

        if (track === "panoptes") {
            flow.push(
                "processing_configure",
                "config_import",
                "config_api_key",
                "config_back_processing"
            );
        }

        flow.push(
            "processing_add_images",
            "raw_select_files",
            "raw_add_selected",
            "processing_start",
            "processing_logs",
            "processing_object",
            "photref_target",
            "photref_image",
            "processing_resume",
            "processing_review_results",
            "results_enter_id",
            "results_apply",
            "complete"
        );

        return flow;
    }

    function defaultState() {
        return {
            version: TUTORIAL_VERSION,
            active: false,
            track: null,
            stepId: null,
        };
    }

    function getState() {
        try {
            const raw = localStorage.getItem(STORAGE_STATE_KEY);
            if (!raw) {
                return defaultState();
            }
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed !== "object") {
                return defaultState();
            }
            if (parsed.version !== TUTORIAL_VERSION) {
                return defaultState();
            }
            return Object.assign(defaultState(), parsed);
        } catch (_error) {
            return defaultState();
        }
    }

    function saveState(state) {
        const value = Object.assign(defaultState(), state);
        try {
            localStorage.setItem(STORAGE_STATE_KEY, JSON.stringify(value));
        } catch (_error) {
            // Ignore storage failures.
        }
    }

    function getPrompted() {
        try {
            return localStorage.getItem(STORAGE_PROMPTED_KEY) === "1";
        } catch (_error) {
            return true;
        }
    }

    function setPrompted() {
        try {
            localStorage.setItem(STORAGE_PROMPTED_KEY, "1");
        } catch (_error) {
            // Ignore storage failures.
        }
    }

    function loadCardPosition() {
        try {
            const raw = localStorage.getItem(STORAGE_CARD_POSITION_KEY);
            if (!raw) {
                return null;
            }
            const parsed = JSON.parse(raw);
            if (
                !parsed ||
                typeof parsed !== "object" ||
                typeof parsed.x !== "number" ||
                typeof parsed.y !== "number"
            ) {
                return null;
            }
            return { x: parsed.x, y: parsed.y };
        } catch (_error) {
            return null;
        }
    }

    function saveCardPosition(position) {
        try {
            localStorage.setItem(
                STORAGE_CARD_POSITION_KEY,
                JSON.stringify(position)
            );
        } catch (_error) {
            // Ignore storage failures.
        }
    }

    function clearCardPosition() {
        manualCardPosition = null;
        try {
            localStorage.removeItem(STORAGE_CARD_POSITION_KEY);
        } catch (_error) {
            // Ignore storage failures.
        }
    }

    function clampCardPosition(card, x, y) {
        const margin = 12;
        const width = card.offsetWidth || 360;
        const height = card.offsetHeight || 250;
        const clampedX = Math.min(
            Math.max(x, margin),
            Math.max(margin, window.innerWidth - width - margin)
        );
        const clampedY = Math.min(
            Math.max(y, margin),
            Math.max(margin, window.innerHeight - height - margin)
        );
        return { x: clampedX, y: clampedY };
    }

    function applyCardPosition(card, x, y) {
        const clamped = clampCardPosition(card, x, y);
        card.style.left = clamped.x + "px";
        card.style.top = clamped.y + "px";
        card.style.right = "";
        card.style.bottom = "";
        return clamped;
    }

    function makeCardDraggable(card, header) {
        header.addEventListener("mousedown", (event) => {
            if (event.button !== 0) {
                return;
            }
            if (event.target.closest("button")) {
                return;
            }

            const rect = card.getBoundingClientRect();
            const offsetX = event.clientX - rect.left;
            const offsetY = event.clientY - rect.top;

            function onMove(moveEvent) {
                manualCardPosition = applyCardPosition(
                    card,
                    moveEvent.clientX - offsetX,
                    moveEvent.clientY - offsetY
                );
            }

            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                if (manualCardPosition) {
                    saveCardPosition(manualCardPosition);
                }
            }

            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
            event.preventDefault();
        });
    }

    function createUI() {
        if (ui) {
            return ui;
        }

        manualCardPosition = loadCardPosition();

        const overlay = document.createElement("div");
        overlay.className = "aw-tutorial-overlay";
        overlay.style.display = "none";

        const card = document.createElement("div");
        card.className = "aw-tutorial-card";
        card.style.display = "none";

        const prompt = document.createElement("div");
        prompt.className = "aw-tutorial-prompt";
        prompt.style.display = "none";

        const fab = document.createElement("button");
        fab.type = "button";
        fab.className = "aw-tutorial-fab";
        fab.textContent = "Tutorial";
        fab.style.display = "none";
        fab.addEventListener("click", openWorkflowPromptFromButton);

        document.body.appendChild(overlay);
        document.body.appendChild(card);
        document.body.appendChild(prompt);
        document.body.appendChild(fab);

        window.addEventListener("resize", () => {
            if (manualCardPosition && card.style.display !== "none") {
                manualCardPosition = applyCardPosition(
                    card,
                    manualCardPosition.x,
                    manualCardPosition.y
                );
                saveCardPosition(manualCardPosition);
            }
        });

        ui = { overlay, card, prompt, fab };
        return ui;
    }

    function clearActionListeners() {
        for (const cleanup of actionCleanup) {
            cleanup();
        }
        actionCleanup = [];
    }

    function clearWaitTimer() {
        if (waitTimerId !== null) {
            window.clearTimeout(waitTimerId);
            waitTimerId = null;
        }
    }

    function clearHighlight() {
        for (const element of highlightedElements) {
            element.classList.remove("aw-tutorial-highlight");
        }
        highlightedElements = [];
    }

    function hideTutorialUI() {
        const { overlay, card, prompt } = createUI();
        overlay.style.display = "none";
        card.style.display = "none";
        prompt.style.display = "none";
        clearHighlight();
        clearWaitTimer();
        clearActionListeners();
    }

    function isHomePage() {
        return window.location.pathname === "/";
    }

    function showFab() {
        const { fab } = createUI();
        const state = getState();
        if (state.track === "sony") {
            fab.textContent = "Tutorial: Sony";
        } else if (state.track === "panoptes") {
            fab.textContent = "Tutorial: PANOPTES";
        } else {
            fab.textContent = "Tutorial";
        }
        fab.style.display = isHomePage() ? "inline-flex" : "none";
    }

    function resolveNewestProjectRow() {
        const rows = Array.from(
            document.querySelectorAll(".tutorial-project-row[data-project-id]")
        );
        if (!rows.length) {
            return [];
        }
        rows.sort((left, right) => {
            const leftId = parseInt(left.getAttribute("data-project-id"), 10) || 0;
            const rightId = parseInt(right.getAttribute("data-project-id"), 10) || 0;
            return rightId - leftId;
        });
        return [rows[0]];
    }

    function resolveTargets(step) {
        if (typeof step.resolveTargets === "function") {
            return step.resolveTargets();
        }

        const targets = [];

        if (Array.isArray(step.selectors)) {
            for (const selector of step.selectors) {
                const element = document.querySelector(selector);
                if (element) {
                    targets.push(element);
                }
            }
            return targets;
        }

        if (step.selector) {
            const element = document.querySelector(step.selector);
            if (element) {
                targets.push(element);
            }
        }

        return targets;
    }

    function resolveActionTargets(step, highlightedTargets) {
        if (typeof step.resolveActionTargets === "function") {
            return step.resolveActionTargets();
        }

        if (step.actionSelector) {
            return Array.from(document.querySelectorAll(step.actionSelector));
        }

        if (step.actionUsesTargets) {
            return highlightedTargets;
        }

        return [];
    }

    function getAnchorTarget(targets) {
        if (!targets.length) {
            return null;
        }
        for (const target of targets) {
            const rect = target.getBoundingClientRect();
            if (
                rect.bottom > 0 &&
                rect.right > 0 &&
                rect.top < window.innerHeight &&
                rect.left < window.innerWidth
            ) {
                return target;
            }
        }
        return targets[0];
    }

    function placeCard(card, target) {
        if (manualCardPosition) {
            manualCardPosition = applyCardPosition(
                card,
                manualCardPosition.x,
                manualCardPosition.y
            );
            return;
        }

        const margin = 12;
        const gap = 14;
        const cardWidth = card.offsetWidth || 360;
        const cardHeight = card.offsetHeight || 250;

        function score(candidate, targetRect) {
            const x1 = candidate.x;
            const y1 = candidate.y;
            const x2 = candidate.x + cardWidth;
            const y2 = candidate.y + cardHeight;

            const overlapW = Math.max(
                0,
                Math.min(x2, targetRect.right) - Math.max(x1, targetRect.left)
            );
            const overlapH = Math.max(
                0,
                Math.min(y2, targetRect.bottom) - Math.max(y1, targetRect.top)
            );
            const overlapArea = overlapW * overlapH;

            const centerDx =
                candidate.x + cardWidth / 2 - (targetRect.left + targetRect.right) / 2;
            const centerDy =
                candidate.y + cardHeight / 2 - (targetRect.top + targetRect.bottom) / 2;
            const distance = Math.sqrt(centerDx * centerDx + centerDy * centerDy);
            return overlapArea * 10000 + distance;
        }

        const fallback = {
            x: window.innerWidth - cardWidth - margin,
            y: margin + 48,
        };

        if (!target) {
            applyCardPosition(card, fallback.x, fallback.y);
            return;
        }

        const targetRect = target.getBoundingClientRect();
        const candidates = [
            { x: targetRect.right + gap, y: targetRect.top },
            { x: targetRect.left - cardWidth - gap, y: targetRect.top },
            { x: targetRect.left, y: targetRect.bottom + gap },
            { x: targetRect.left, y: targetRect.top - cardHeight - gap },
            fallback,
            { x: window.innerWidth - cardWidth - margin, y: window.innerHeight - cardHeight - margin },
        ].map((candidate) => clampCardPosition(card, candidate.x, candidate.y));

        let best = candidates[0];
        let bestScore = score(best, targetRect);
        for (let index = 1; index < candidates.length; index += 1) {
            const candidate = candidates[index];
            const candidateScore = score(candidate, targetRect);
            if (candidateScore < bestScore) {
                best = candidate;
                bestScore = candidateScore;
            }
        }

        applyCardPosition(card, best.x, best.y);
    }

    function isTransientPath(pathname) {
        return TRANSIENT_PATHS.some((pattern) => pattern.test(pathname));
    }

    function stepIndex(flow, stepId) {
        return flow.indexOf(stepId);
    }

    function promoteForCurrentPath(state, flow) {
        if (!state.active || !state.stepId) {
            return state;
        }

        const currentPath = window.location.pathname;
        const currentIndex = stepIndex(flow, state.stepId);

        if (currentIndex < 0) {
            state.stepId = flow[0] || null;
            return state;
        }

        const currentStep = STEPS[state.stepId];
        if (currentStep && currentStep.path.test(currentPath)) {
            return state;
        }

        for (let index = currentIndex + 1; index < flow.length; index += 1) {
            const candidateId = flow[index];
            const candidate = STEPS[candidateId];
            if (candidate.path.test(currentPath)) {
                state.stepId = candidateId;
                return state;
            }
        }

        return state;
    }

    function finishTutorial() {
        const current = getState();
        saveState({
            version: TUTORIAL_VERSION,
            active: false,
            track: current.track || null,
            stepId: null,
        });
        hideTutorialUI();
        showFab();
    }

    function setStep(stepId) {
        const state = getState();
        if (!state.active || !state.track) {
            return;
        }
        saveState({
            version: TUTORIAL_VERSION,
            active: true,
            track: state.track,
            stepId: stepId,
        });
        renderTutorial();
    }

    function bindAction(step, flow, currentStepId, highlightedTargets) {
        const actionTargets = resolveActionTargets(step, highlightedTargets);
        if (!actionTargets.length) {
            return;
        }

        const currentIndex = stepIndex(flow, currentStepId);
        const nextId =
            currentIndex >= 0 && currentIndex < flow.length - 1
                ? flow[currentIndex + 1]
                : null;

        for (const target of actionTargets) {
            const handler = () => {
                if (!nextId) {
                    return;
                }
                const state = getState();
                if (!state.active || !state.track) {
                    return;
                }
                saveState({
                    version: TUTORIAL_VERSION,
                    active: true,
                    track: state.track,
                    stepId: nextId,
                });
            };
            target.addEventListener("click", handler, true);
            actionCleanup.push(() => {
                target.removeEventListener("click", handler, true);
            });
        }
    }

    function startTutorial(track) {
        setPrompted();
        saveState({
            version: TUTORIAL_VERSION,
            active: true,
            track: track,
            stepId: "home_new_project",
        });

        const { prompt } = createUI();
        prompt.style.display = "none";

        if (!isHomePage()) {
            window.location.assign("/");
            return;
        }

        renderTutorial();
    }

    function openFirstVisitPrompt() {
        const { prompt, card, overlay } = createUI();
        card.style.display = "none";
        overlay.style.display = "none";

        prompt.innerHTML = "";

        const title = document.createElement("h2");
        title.textContent = "Welcome to AutoWISP BUI";

        const text = document.createElement("p");
        text.textContent =
            "Would you like a guided walkthrough to create your first light curve?";

        const actions = document.createElement("div");
        actions.className = "aw-tutorial-actions";

        const startButton = document.createElement("button");
        startButton.type = "button";
        startButton.className = "aw-tutorial-primary";
        startButton.textContent = "Start Tutorial";
        startButton.addEventListener("click", () => {
            setPrompted();
            const state = getState();
            if (state.track) {
                startTutorial(state.track);
            } else {
                openWorkflowPrompt();
            }
        });

        const skipButton = document.createElement("button");
        skipButton.type = "button";
        skipButton.className = "aw-tutorial-secondary";
        skipButton.textContent = "Skip For Now";
        skipButton.addEventListener("click", () => {
            setPrompted();
            prompt.style.display = "none";
        });

        actions.appendChild(startButton);
        actions.appendChild(skipButton);

        prompt.appendChild(title);
        prompt.appendChild(text);
        prompt.appendChild(actions);
        prompt.style.display = "block";
    }

    function openWorkflowPrompt() {
        const { prompt, card, overlay } = createUI();
        card.style.display = "none";
        overlay.style.display = "none";

        prompt.innerHTML = "";

        const title = document.createElement("h2");
        title.textContent = "Choose Tutorial Workflow";

        const text = document.createElement("p");
        text.textContent =
            "Sony and PANOPTES use different setup order. Select one to continue.";

        const actions = document.createElement("div");
        actions.className = "aw-tutorial-actions";

        const sonyButton = document.createElement("button");
        sonyButton.type = "button";
        sonyButton.className = "aw-tutorial-primary";
        sonyButton.textContent = "Sony";
        sonyButton.addEventListener("click", () => startTutorial("sony"));

        const panoptesButton = document.createElement("button");
        panoptesButton.type = "button";
        panoptesButton.className = "aw-tutorial-primary";
        panoptesButton.textContent = "PANOPTES";
        panoptesButton.addEventListener("click", () =>
            startTutorial("panoptes")
        );

        const cancelButton = document.createElement("button");
        cancelButton.type = "button";
        cancelButton.className = "aw-tutorial-secondary";
        cancelButton.textContent = "Cancel";
        cancelButton.addEventListener("click", () => {
            if (getState().active) {
                renderTutorial();
            } else {
                prompt.style.display = "none";
            }
        });

        actions.appendChild(sonyButton);
        actions.appendChild(panoptesButton);
        actions.appendChild(cancelButton);

        prompt.appendChild(title);
        prompt.appendChild(text);
        prompt.appendChild(actions);
        prompt.style.display = "block";
    }

    function openWorkflowPromptFromButton() {
        const state = getState();
        if (state.track) {
            startTutorial(state.track);
        } else {
            openWorkflowPrompt();
        }
    }

    function buildCard(step, flow, currentStepId, pathMismatch, targetMissing) {
        const { card, prompt, overlay } = createUI();

        prompt.style.display = "none";
        overlay.style.display = "none";
        card.style.display = "block";
        card.innerHTML = "";

        const currentIndex = stepIndex(flow, currentStepId);
        const header = document.createElement("div");
        header.className = "aw-tutorial-card-header";

        const stepCount = document.createElement("div");
        stepCount.className = "aw-tutorial-step-count";
        stepCount.textContent =
            "Step " +
            (currentIndex + 1).toString() +
            " of " +
            flow.length.toString();

        const headerTools = document.createElement("div");
        headerTools.className = "aw-tutorial-header-tools";

        const autoPlaceButton = document.createElement("button");
        autoPlaceButton.type = "button";
        autoPlaceButton.className = "aw-tutorial-secondary aw-tutorial-mini";
        autoPlaceButton.textContent = "Auto";
        autoPlaceButton.title = "Auto place the tutorial box";
        autoPlaceButton.addEventListener("click", () => {
            clearCardPosition();
            renderTutorial();
        });

        const exitButton = document.createElement("button");
        exitButton.type = "button";
        exitButton.className = "aw-tutorial-secondary aw-tutorial-mini";
        exitButton.textContent = "Exit";
        exitButton.title = "End tutorial";
        exitButton.addEventListener("click", () => {
            const shouldExit = window.confirm(
                "End the tutorial now? You can restart it from the Tutorial button."
            );
            if (shouldExit) {
                finishTutorial();
            }
        });

        headerTools.appendChild(autoPlaceButton);
        headerTools.appendChild(exitButton);
        header.appendChild(stepCount);
        header.appendChild(headerTools);

        const title = document.createElement("h3");
        title.textContent = step.title;

        const text = document.createElement("p");
        if (pathMismatch && step.goTo) {
            text.textContent =
                "You are on a different page. Jump to the expected page to continue this step.";
        } else if (targetMissing && step.waitForTarget) {
            text.textContent =
                "Waiting for this control to appear. Continue processing and the tutorial will continue automatically.";
        } else if (targetMissing) {
            text.textContent =
                "This control is not visible yet. You can continue with Next or go to the expected page.";
        } else {
            text.textContent = step.body;
        }

        const actions = document.createElement("div");
        actions.className = "aw-tutorial-actions";

        if (currentIndex > 0) {
            const backButton = document.createElement("button");
            backButton.type = "button";
            backButton.className = "aw-tutorial-secondary";
            backButton.textContent = "Back";
            backButton.addEventListener("click", () => {
                setStep(flow[currentIndex - 1]);
            });
            actions.appendChild(backButton);
        }

        if (pathMismatch && step.goTo) {
            const goButton = document.createElement("button");
            goButton.type = "button";
            goButton.className = "aw-tutorial-primary";
            goButton.textContent = "Go To Step";
            goButton.addEventListener("click", () => {
                window.location.assign(step.goTo);
            });
            actions.appendChild(goButton);
        } else if (step.isTerminal) {
            const doneButton = document.createElement("button");
            doneButton.type = "button";
            doneButton.className = "aw-tutorial-primary";
            doneButton.textContent = "Finish";
            doneButton.addEventListener("click", finishTutorial);
            actions.appendChild(doneButton);
        } else if (currentIndex < flow.length - 1) {
            const nextButton = document.createElement("button");
            nextButton.type = "button";
            nextButton.className = "aw-tutorial-primary";
            nextButton.textContent = "Next";
            nextButton.addEventListener("click", () => {
                setStep(flow[currentIndex + 1]);
            });
            actions.appendChild(nextButton);
        }

        if (currentIndex === 0) {
            const switchWorkflowButton = document.createElement("button");
            switchWorkflowButton.type = "button";
            switchWorkflowButton.className = "aw-tutorial-secondary";
            switchWorkflowButton.textContent = "Switch Workflow";
            switchWorkflowButton.addEventListener("click", openWorkflowPrompt);
            actions.appendChild(switchWorkflowButton);
        }

        card.appendChild(header);
        card.appendChild(title);
        card.appendChild(text);
        card.appendChild(actions);
        makeCardDraggable(card, header);
    }

    function renderTutorial() {
        const { prompt } = createUI();
        showFab();

        let state = getState();

        if (!state.active) {
            hideTutorialUI();
            showFab();
            if (!getPrompted() && isHomePage()) {
                openFirstVisitPrompt();
            } else {
                prompt.style.display = "none";
            }
            return;
        }

        if (!state.track) {
            finishTutorial();
            return;
        }

        const flow = getFlow(state.track);

        if (!state.stepId || stepIndex(flow, state.stepId) < 0) {
            state.stepId = flow[0];
        }

        state = promoteForCurrentPath(state, flow);
        saveState(state);

        clearActionListeners();
        clearWaitTimer();
        clearHighlight();

        const step = STEPS[state.stepId];
        if (!step) {
            finishTutorial();
            return;
        }

        const pathname = window.location.pathname;
        const pathMatches = step.path.test(pathname);

        if (!pathMatches && isTransientPath(pathname)) {
            waitTimerId = window.setTimeout(renderTutorial, 500);
            return;
        }

        let targets = [];
        if (pathMatches) {
            targets = resolveTargets(step);
            for (const target of targets) {
                target.classList.add("aw-tutorial-highlight");
            }
            highlightedElements = targets.slice();
        }

        const expectsTarget =
            step.selector || step.selectors || step.resolveTargets;
        const targetMissing =
            pathMatches && Boolean(expectsTarget) && targets.length === 0;
        const pathMismatch = !pathMatches;

        buildCard(step, flow, state.stepId, pathMismatch, targetMissing);
        placeCard(createUI().card, getAnchorTarget(targets));

        if (targetMissing && step.waitForTarget) {
            waitTimerId = window.setTimeout(renderTutorial, 1500);
        }

        if (pathMatches) {
            bindAction(step, flow, state.stepId, targets);
        }
    }

    createUI();
    renderTutorial();
})();
