// Global State
let modelMetrics = null;

// Mock Examples
const samples = {
    spam: {
        ham: "Hey! Just checking if we are still meeting up for lunch today at 12:30. Let me know if you need a ride.",
        spam: "URGENT! Your mobile number has been selected for a cash prize of £2,000. Call 09061701461 now! Claim code: KL341."
    },
    tweets: {
        normal: "The local community garden is looking absolutely beautiful this morning. So grateful for all the volunteers!",
        offensive: "That referee is a dumb piece of shit, honestly how do you make a call that stupid? Complete garbage.",
        hate: "We must protect our land from these invaders. They are animals and don't belong in our civilized nation."
    }
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    // Clock setup
    updateClock();
    setInterval(updateClock, 30000);

    // Tab Navigation
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", (e) => {
            const tabId = e.currentTarget.getAttribute("data-tab");
            switchTab(tabId);
        });
    });

    // Fetch metrics from API
    fetchMetrics();
});

// Update top header clock
function updateClock() {
    const clock = document.getElementById("current-time");
    if (clock) {
        const now = new Date();
        const hrs = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        clock.textContent = `${hrs}:${mins}`;
    }
}

// Switch between main panels
function switchTab(tabId) {
    // Update sidebar links active class
    document.querySelectorAll(".nav-item").forEach(item => {
        if (item.getAttribute("data-tab") === tabId) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    // Update main panel visibility
    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
    });
    
    const activePane = document.getElementById(`tab-${tabId}`);
    if (activePane) {
        activePane.classList.add("active");
    }

    // Set page headers
    const title = document.getElementById("page-title");
    const subtitle = document.getElementById("page-subtitle");
    
    if (tabId === "dashboard") {
        title.textContent = "Project Overview";
        subtitle.textContent = "Interactive analysis and metrics dashboard for NLP models";
    } else if (tabId === "spam-classifier") {
        title.textContent = "Spam Sentinel";
        subtitle.textContent = "Real-time SMS spam-ham classification engine";
    } else if (tabId === "tweet-classifier") {
        title.textContent = "Tweet Guard";
        subtitle.textContent = "Social media safety moderation and classification";
    } else if (tabId === "analytics") {
        title.textContent = "Model Analytics";
        subtitle.textContent = "Complete performance reports, feature reductions, and confusion matrices";
        // Render charts once analytics tab is shown
        if (modelMetrics) {
            renderAnalyticsViews();
        }
    }
}

// Switch between sub panes of Analytics tab
function switchSubAnalytics(subPaneId) {
    document.querySelectorAll(".tabs-sub button").forEach(btn => {
        btn.classList.remove("sub-tabactive");
    });
    document.getElementById(`btn-sub-${subPaneId}`).classList.add("sub-tabactive");

    document.querySelectorAll(".sub-pane").forEach(pane => {
        pane.classList.remove("active");
    });
    document.getElementById(`sub-pane-${subPaneId}`).classList.add("active");
}

// Fetch Metrics from backend
async function fetchMetrics() {
    try {
        const response = await fetch("/api/metrics");
        if (!response.ok) throw new Error("Failed to load metrics");
        modelMetrics = await response.json();
        
        // Update overview tab labels
        document.getElementById("spam-acc-lbl").textContent = `${(modelMetrics.spam.accuracy * 100).toFixed(2)}%`;
        document.getElementById("tweet-acc-lbl").textContent = `${(modelMetrics.tweets.accuracy * 100).toFixed(2)}%`;
    } catch (error) {
        console.error("Error loading metrics:", error);
    }
}

// Render the analytics charts, tables, matrices
function renderAnalyticsViews() {
    if (!modelMetrics) return;

    // --- RENDER SPAM ANALYTICS ---
    // Classification Report Table
    const spamReportBody = document.getElementById("spam-report-body");
    spamReportBody.innerHTML = "";
    const spamRep = modelMetrics.spam.classification_report;
    const spamClasses = ["ham", "spam"];
    
    spamClasses.forEach(cls => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><span class="badge ${cls === 'spam' ? 'spam-badge' : 'spam-badge'}">${cls}</span></td>
            <td>${(spamRep[cls].precision * 100).toFixed(1)}%</td>
            <td>${(spamRep[cls].recall * 100).toFixed(1)}%</td>
            <td>${(spamRep[cls]["f1-score"] * 100).toFixed(1)}%</td>
            <td>${spamRep[cls].support}</td>
        `;
        spamReportBody.appendChild(row);
    });
    
    // Add overall row
    const spamTotalRow = document.createElement("tr");
    spamTotalRow.className = "total-row";
    spamTotalRow.innerHTML = `
        <td>accuracy</td>
        <td></td>
        <td></td>
        <td>${(spamRep.accuracy * 100).toFixed(1)}%</td>
        <td>${spamRep.macro.support || spamRep["macro avg"].support}</td>
    `;
    spamReportBody.appendChild(spamTotalRow);

    // Confusion Matrix Grid
    const spamMatrix = document.getElementById("spam-matrix-grid");
    spamMatrix.innerHTML = "";
    spamMatrix.style.gridTemplateColumns = "repeat(3, 1fr)";
    
    // Labels
    spamMatrix.appendChild(createLabelCell(""));
    spamMatrix.appendChild(createLabelCell("Pred Ham"));
    spamMatrix.appendChild(createLabelCell("Pred Spam"));
    
    const spamCM = modelMetrics.spam.confusion_matrix;
    // Row 0: Ham
    spamMatrix.appendChild(createLabelCell("Act Ham"));
    spamMatrix.appendChild(createCell(spamCM[0][0], true, "TN"));
    spamMatrix.appendChild(createCell(spamCM[0][1], false, "FP"));
    // Row 1: Spam
    spamMatrix.appendChild(createLabelCell("Act Spam"));
    spamMatrix.appendChild(createCell(spamCM[1][0], false, "FN"));
    spamMatrix.appendChild(createCell(spamCM[1][1], true, "TP"));

    // Top words
    renderTopWordsList("spam-top-ham-list", modelMetrics.spam.top_ham_features);
    renderTopWordsList("spam-top-spam-list", modelMetrics.spam.top_spam_features);


    // --- RENDER TWEET ANALYTICS ---
    // Classification Report Table
    const tweetsReportBody = document.getElementById("tweets-report-body");
    tweetsReportBody.innerHTML = "";
    const tweetRep = modelMetrics.tweets.classification_report;
    const tweetClasses = ["normal", "offensive", "hate_speech"];
    const tweetKeyMap = { "normal": "normal", "offensive": "offensive", "hate_speech": "hate_speech" };
    
    tweetClasses.forEach(cls => {
        const data = tweetRep[cls];
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><span class="${cls}-text font-weight-bold" style="text-transform: capitalize;">${cls.replace('_', ' ')}</span></td>
            <td>${(data.precision * 100).toFixed(1)}%</td>
            <td>${(data.recall * 100).toFixed(1)}%</td>
            <td>${(data["f1-score"] * 100).toFixed(1)}%</td>
            <td>${data.support}</td>
        `;
        tweetsReportBody.appendChild(row);
    });

    const tweetTotalRow = document.createElement("tr");
    tweetTotalRow.className = "total-row";
    tweetTotalRow.innerHTML = `
        <td>accuracy</td>
        <td></td>
        <td></td>
        <td>${(tweetRep.accuracy * 100).toFixed(1)}%</td>
        <td>${tweetRep.macro.support || tweetRep["macro avg"].support}</td>
    `;
    tweetsReportBody.appendChild(tweetTotalRow);

    // Confusion Matrix Grid
    const tweetsMatrix = document.getElementById("tweets-matrix-grid");
    tweetsMatrix.innerHTML = "";
    tweetsMatrix.style.gridTemplateColumns = "repeat(4, 1fr)";
    
    tweetsMatrix.appendChild(createLabelCell(""));
    tweetsMatrix.appendChild(createLabelCell("Pred Hate"));
    tweetsMatrix.appendChild(createLabelCell("Pred Offen"));
    tweetsMatrix.appendChild(createLabelCell("Pred Norm"));
    
    const tweetCM = modelMetrics.tweets.confusion_matrix;
    const labels = ["Act Hate", "Act Offen", "Act Norm"];
    for (let r = 0; r < 3; r++) {
        tweetsMatrix.appendChild(createLabelCell(labels[r]));
        for (let c = 0; c < 3; c++) {
            const isCorrect = r === c;
            tweetsMatrix.appendChild(createCell(tweetCM[r][c], isCorrect, isCorrect ? "Correct" : "Error"));
        }
    }

    // Update reduction banner numbers
    document.getElementById("feat-before").textContent = modelMetrics.tweets.features_before.toLocaleString();
    document.getElementById("feat-after").textContent = modelMetrics.tweets.features_after.toLocaleString();

    // Top words per class
    renderTopWordsList("tweets-top-normal-list", modelMetrics.tweets.top_features.normal);
    renderTopWordsList("tweets-top-offensive-list", modelMetrics.tweets.top_features.offensive);
    renderTopWordsList("tweets-top-hate-list", modelMetrics.tweets.top_features.hate_speech);
}

// Confusion matrix helpers
function createLabelCell(text) {
    const el = document.createElement("div");
    el.className = "matrix-axis-lbl";
    el.textContent = text;
    el.style.display = "flex";
    el.style.justifyContent = "center";
    el.style.alignEnabled = "center";
    el.style.alignItems = "center";
    return el;
}

function createCell(value, isCorrect, title) {
    const el = document.createElement("div");
    el.className = `matrix-cell ${isCorrect ? 'correct' : 'error'}`;
    el.innerHTML = `
        <span class="val">${value}</span>
        <span class="lbl">${title}</span>
    `;
    return el;
}

// Render Lists of top words
function renderTopWordsList(containerId, wordsArray) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    if (!wordsArray) return;
    
    wordsArray.forEach(item => {
        const row = document.createElement("div");
        row.className = "word-row-item";
        row.innerHTML = `
            <span class="word">${item.word}</span>
            <span class="coef">${item.importance.toFixed(3)}</span>
        `;
        container.appendChild(row);
    });
}

// Insert mock samples
function insertSpamSample(type) {
    document.getElementById("spam-input").value = samples.spam[type];
}

function insertTweetSample(type) {
    document.getElementById("tweet-input").value = samples.tweets[type];
}

// REST API calls for classification
async function runSpamClassification() {
    const input = document.getElementById("spam-input").value.trim();
    if (!input) return;

    const btn = document.getElementById("spam-submit-btn");
    const resultCard = document.getElementById("spam-result-card");
    const analysisEmpty = document.getElementById("spam-analysis-empty");
    const analysisList = document.getElementById("spam-analysis-list");
    
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/classify/spam", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: input })
        });
        
        if (!response.ok) throw new Error("Classification failed");
        
        const data = await response.json();
        
        // Show result
        resultCard.classList.remove("hidden");
        const badge = document.getElementById("spam-result-badge");
        badge.className = `result-badge ${data.class}`;
        badge.textContent = data.class;
        
        const confPercent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById("spam-confidence-text").textContent = confPercent;
        document.getElementById("spam-confidence-fill").style.width = confPercent;
        
        // Show contributing words
        analysisEmpty.classList.add("hidden");
        analysisList.classList.remove("hidden");
        analysisList.innerHTML = "";
        
        if (data.words_contributed && data.words_contributed.length > 0) {
            data.words_contributed.forEach(item => {
                const row = document.createElement("div");
                const isSpamShift = item.spam_factor > 0;
                row.className = `feature-bar-row ${isSpamShift ? 'spam-shift' : 'ham-shift'}`;
                
                row.innerHTML = `
                    <span class="word">${item.word}</span>
                    <div class="weight-bar-box">
                        <span class="weight-lbl">${isSpamShift ? '+' : ''}${item.spam_factor.toFixed(2)}</span>
                        <span class="badge ${isSpamShift ? 'spam-badge' : 'spam-badge'}" style="font-size: 0.6rem; padding: 2px 6px;">
                            ${isSpamShift ? 'spam' : 'ham'}
                        </span>
                    </div>
                `;
                analysisList.appendChild(row);
            });
        } else {
            analysisList.innerHTML = `<div class="empty-state"><p>No clear vocabulary indicators found in this message.</p></div>`;
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred during classification.");
    } finally {
        btn.disabled = false;
        btn.textContent = "Classify Message";
    }
}

async function runTweetClassification() {
    const input = document.getElementById("tweet-input").value.trim();
    if (!input) return;

    const btn = document.getElementById("tweet-submit-btn");
    const resultCard = document.getElementById("tweet-result-card");
    const analysisEmpty = document.getElementById("tweet-analysis-empty");
    const analysisList = document.getElementById("tweet-analysis-list");
    
    btn.disabled = true;
    btn.textContent = "Moderating...";

    try {
        const response = await fetch("/api/classify/tweet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweet: input })
        });
        
        if (!response.ok) throw new Error("Moderation failed");
        
        const data = await response.json();
        
        // Show result
        resultCard.classList.remove("hidden");
        const badge = document.getElementById("tweet-result-badge");
        badge.className = `result-badge ${data.class}`;
        badge.textContent = data.class.replace('_', ' ');
        
        const confPercent = `${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById("tweet-confidence-text").textContent = confPercent;
        document.getElementById("tweet-confidence-fill").style.width = confPercent;
        
        // Update multiclass probabilities
        const normProb = `${(data.all_confidences.normal * 100).toFixed(1)}%`;
        const offProb = `${(data.all_confidences.offensive * 100).toFixed(1)}%`;
        const hateProb = `${(data.all_confidences.hate_speech * 100).toFixed(1)}%`;

        document.getElementById("prob-normal-bar").style.width = normProb;
        document.getElementById("prob-normal-val").textContent = normProb;

        document.getElementById("prob-offensive-bar").style.width = offProb;
        document.getElementById("prob-offensive-val").textContent = offProb;

        document.getElementById("prob-hate-bar").style.width = hateProb;
        document.getElementById("prob-hate-val").textContent = hateProb;

        // Show contributing words
        analysisEmpty.classList.add("hidden");
        analysisList.classList.remove("hidden");
        analysisList.innerHTML = "";
        
        if (data.words_contributed && data.words_contributed.length > 0) {
            data.words_contributed.forEach(item => {
                const row = document.createElement("div");
                row.className = `feature-bar-row pos-weight`;
                
                row.innerHTML = `
                    <span class="word">${item.word}</span>
                    <div class="weight-bar-box">
                        <span class="weight-lbl">${item.weight.toFixed(3)}</span>
                    </div>
                `;
                analysisList.appendChild(row);
            });
        } else {
            analysisList.innerHTML = `<div class="empty-state"><p>No vocabulary weights found for this class.</p></div>`;
        }

    } catch (error) {
        console.error(error);
        alert("An error occurred during moderation.");
    } finally {
        btn.disabled = false;
        btn.textContent = "Moderate Tweet";
    }
}
