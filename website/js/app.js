/**
 * Grant's Election Forecast - Frontend Application
 */

// Global state
let houseData = null;
let senateData = null;
let districtGeoJSON = null;
let statesGeoJSON = null;
let map = null;
let geoLayer = null;
let currentChamber = 'house';
let seatChart = null;
let timelineChart = null;
let nationalTrendChart = null;
let houseTimeline = null;
let senateTimeline = null;
let currentTimelineChamber = 'house';
let sortColumn = 'prob_dem';
let sortDirection = 'desc';

const BASEMAP = {
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
};

// Color scheme for ratings
const COLORS = {
    safe_d: '#0015BC',
    likely_d: '#5580E0',
    lean_d: '#A8C8F0',
    toss_up: '#888888',
    lean_r: '#F0A8A8',
    likely_r: '#E05555',
    safe_r: '#BC0000',
    guaranteed_d: '#000D7A',  // Darker blue for seats not up
    guaranteed_r: '#7A0000',  // Darker red for seats not up
};

// FIPS to state abbreviation mapping
const FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "72": "PR"
};

// Parse CSV data
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].trim().split(',');
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const values = lines[i].trim().split(',');
        const row = {};
        headers.forEach((header, idx) => {
            const val = values[idx];
            // Keep date column as string, parse others as numbers
            if (header === 'date') {
                row[header] = val;
            } else {
                const num = parseFloat(val);
                row[header] = isNaN(num) ? val : num;
            }
        });
        data.push(row);
    }

    return data;
}

// Each optional resource can fail without hiding the rest of the archive.
async function loadResource(path, format = 'json') {
    try {
        const response = await fetch(`${path}?v=archive-1`);
        if (!response.ok) return null;
        return format === 'csv' ? parseCSV(await response.text()) : await response.json();
    } catch (error) {
        console.warn(`Could not load ${path}`, error);
        return null;
    }
}

async function loadData() {
    [houseData, senateData, districtGeoJSON, statesGeoJSON, houseTimeline, senateTimeline] = await Promise.all([
        loadResource('./forecast.json'), loadResource('./senate_forecast.json'),
        loadResource('./data/districts.geojson'), loadResource('./data/states.geojson'),
        loadResource('./timeline.csv', 'csv'), loadResource('./senate_timeline.csv', 'csv'),
    ]);
    if (!houseData && !senateData) {
        const error = document.getElementById('forecast-error');
        error.hidden = false;
        error.textContent = 'The archived forecast data could not be loaded. Please reload the page.';
        return;
    }
    initializePage();
}

function componentMessage(id, message = '') {
    const target = document.getElementById(id);
    if (!target) return;
    let status = document.querySelector(`[data-status-for="${id}"]`);
    if (!status) {
        status = document.createElement('p');
        status.dataset.statusFor = id;
        status.className = 'component-message';
        status.setAttribute('role', 'status');
        target.before(status);
    }
    status.textContent = message;
    status.hidden = !message;
    target.hidden = Boolean(message);
    if (target.tagName === 'CANVAS') target.parentElement?.classList.toggle('unavailable', Boolean(message));
}

function renderOptional(id, action) {
    try {
        action();
    } catch (error) {
        console.warn(`Could not display ${id}`, error);
        componentMessage(id, 'This visualization is unavailable. The archived statistics and race table remain available.');
    }
}

function chartAvailable(id) {
    if (typeof Chart !== 'function') {
        componentMessage(id, 'Chart unavailable. The archived statistics and race table remain available.');
        return false;
    }
    componentMessage(id);
    return true;
}

function renderMap() {
    const geometry = currentChamber === 'house' ? districtGeoJSON : statesGeoJSON;
    if (typeof L === 'undefined' || !geometry) {
        componentMessage('map-container', 'Map unavailable. Explore the archived races in the table below.');
        return;
    }
    componentMessage('map-container');
    if (!map) initializeMap();
    else {
        updateMapLayer();
        map.invalidateSize();
    }
}

// Calendar dates use UTC consistently, independent of the visitor's timezone.
function dateValue(date) {
    return Date.parse(`${date}T00:00:00Z`);
}

function chartDate(value) {
    return new Date(value).toLocaleDateString('en-US', {month: 'short', day: 'numeric', timeZone: 'UTC'});
}

function dateAxis(points) {
    const dates = points.map(point => point.x);
    return {
        type: 'linear', min: Math.min(...dates), max: Math.max(...dates),
        title: {display: true, text: 'Date', color: '#aaa'},
        ticks: {color: '#aaa', maxTicksLimit: 6, callback: chartDate},
        grid: {display: false},
    };
}

function initializePage() {
    if (houseData) updateHouseStats();
    if (senateData) updateSenateStats();
    document.querySelectorAll('.chamber-btn').forEach(button => {
        button.disabled = !(button.dataset.chamber === 'house' ? houseData : senateData);
        if (button.disabled) button.title = 'Archived chamber data unavailable';
    });
    if (!houseData || !senateData) {
        const error = document.getElementById('forecast-error');
        error.hidden = false;
        error.textContent = `${houseData ? 'Senate' : 'House'} archive data is unavailable. The other chamber remains available.`;
    }
    if (houseData) {
        updateModelCard();
        renderOptional('national-trend-chart', createNationalTrendChart);
    } else {
        componentMessage('polling-model', 'The archived House polling explanation is unavailable.');
    }
    setupEventListeners();
    selectChamber(houseData ? 'house' : 'senate');
    currentTimelineChamber = houseTimeline?.length ? 'house' : 'senate';
    document.querySelectorAll('.timeline-btn').forEach(button => {
        button.classList.toggle('active', button.dataset.timeline === currentTimelineChamber);
    });
    renderOptional('timeline-chart', createTimelineChart);
}

function selectChamber(chamber) {
    const data = chamber === 'house' ? houseData : senateData;
    if (!data) return;
    currentChamber = chamber;
    document.querySelectorAll('.chamber-btn').forEach(button => {
        button.classList.toggle('active', button.dataset.chamber === chamber);
        button.setAttribute('aria-pressed', String(button.dataset.chamber === chamber));
    });
    document.querySelectorAll('.chamber-section').forEach(section => {
        section.classList.toggle('active', section.dataset.chamber === chamber);
    });
    renderNationalEnvironment(data, chamber);
    renderOptional('map-container', renderMap);
    updateMapLegend();
    renderOptional('seat-chart', createSeatChart);
    populateStateFilter();
    document.getElementById('category-filter').value = '';
    document.getElementById('search-filter').value = '';
    closeModal();
    renderTable();
}

// Update House statistics
function updateHouseStats() {
    const { summary, metadata, categories } = houseData;

    // Update time
    const updateTime = new Date(metadata.updated_at);
    document.getElementById('update-time').textContent = formatDate(updateTime);

    // Stats
    document.getElementById('house-dem-prob').textContent = `${Math.round(summary.prob_dem_majority * 100)}%`;
    document.getElementById('house-median-seats').textContent = summary.median_dem_seats;
    document.getElementById('house-confidence-interval').textContent = `${summary.ci_90_low}-${summary.ci_90_high}`;

    // Categories bar
    updateCategoriesBar('house-categories-bar', categories, 435);
}

function formatMargin(value) {
    const number = Number(value || 0);
    return number >= 0 ? `D+${number.toFixed(1)}` : `R+${Math.abs(number).toFixed(1)}`;
}

function formatInterval(interval) {
    if (!Array.isArray(interval) || interval.length !== 2) return '90% interval unavailable';
    return `90% interval ${formatMargin(interval[0])} to ${formatMargin(interval[1])}`;
}

function formatEconomicChange(value, unit) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    const suffix = unit === 'percent' ? '%' : unit === 'percentage points' ? ' pp' : ' pts';
    return `${number >= 0 ? '+' : ''}${number.toFixed(2)}${suffix}`;
}

function renderNationalEnvironment(data, chamber) {
    if (!data?.national_environment) {
        componentMessage('national-environment', 'Archived national environment data is unavailable for this chamber.');
        return;
    }
    componentMessage('national-environment');
    const env = data.national_environment;
    const prior = env.fundamentals_prior || {};
    const published = env.published_sentiment || {};
    const polling = env.polling_input || {};
    const current = env.poll_updated_current || {};
    const election = env.election_day || {};
    const economy = env.economy || {};
    const chamberName = chamber === 'house' ? 'House' : 'Senate';

    document.getElementById('national-chamber-label').textContent = chamberName;
    document.getElementById('ne-polling-label').textContent = `${chamberName} polling input`;
    document.getElementById('ne-fundamentals').textContent = formatMargin(prior.mean);
    document.getElementById('ne-fundamentals-detail').textContent = `${formatInterval(prior.ci_90)} · σ ${Number(prior.std || 0).toFixed(2)}`;
    document.getElementById('ne-published').textContent = formatMargin(published.margin);
    document.getElementById('ne-published-detail').textContent = `${published.date || '—'} · Silver likely-voter average`;
    document.getElementById('ne-polling').textContent = formatMargin(polling.margin);
    document.getElementById('ne-polling-detail').textContent = `${polling.date || '—'} · σ ${Number(polling.std || 0).toFixed(2)} · ${polling.poll_rows || 0} ${polling.poll_rows === 1 ? 'input' : 'weighted rows'}`;
    document.getElementById('ne-current').textContent = formatMargin(current.mean);
    document.getElementById('ne-current-detail').textContent = `${current.date || '—'} · ${formatInterval(current.ci_90)}`;
    document.getElementById('ne-election').textContent = formatMargin(election.mean);
    document.getElementById('ne-election-detail').textContent = `${election.date || '—'} · ${formatInterval(election.ci_90)} · σ ${Number(election.std || 0).toFixed(2)}`;
    document.getElementById('ne-economy').textContent = `${Number(economy.standardized_index || 0) >= 0 ? '+' : ''}${Number(economy.standardized_index || 0).toFixed(2)} SD`;
    document.getElementById('ne-economy-detail').textContent = `${economy.interpretation || '—'} · calculated ${economy.calculation_date || '—'}`;
    document.getElementById('ne-archive-date').textContent = chartDate(dateValue(data.metadata.election_date)) + ', 2026';

    document.getElementById('economic-components').innerHTML = Object.values(economy.components || {}).map(component => `
        <tr>
            <td><strong>${component.label || 'Economic metric'}</strong></td>
            <td>${formatEconomicChange(component.model_oriented_change, component.unit)}</td>
            <td>${(Number(component.weight || 0) * 100).toFixed(0)}%</td>
            <td>${Number(component.contribution) >= 0 ? '+' : ''}${Number(component.contribution || 0).toFixed(2)}</td>
            <td>${component.observation_date || '—'}</td>
        </tr>`).join('');
}

function updateModelCard() {
    const model = houseData.national_model || {};
    document.getElementById('prior-margin').textContent = formatMargin(model.prior?.mean);
    document.getElementById('current-margin').textContent = formatMargin(model.current_sentiment?.mean);
    document.getElementById('election-margin').textContent = formatMargin(model.election_day?.mean);

    const published = houseData.polling?.published_average || {};
    const likelihood = houseData.polling?.national_likelihood || {};
    document.getElementById('likelihood-summary').innerHTML = `
        <div><span>Silver published average</span><strong>${formatMargin(published.margin)}</strong><small>${published.date || '—'} · likely-voter adjusted</small></div>
        <div><span>House model likelihood</span><strong>${formatMargin(likelihood.margin)}</strong><small>${likelihood.date || '—'} · ${likelihood.poll_rows || 0} weighted poll rows · σ ${Number(likelihood.observation_std || 0).toFixed(2)}</small></div>`;
    document.getElementById('method-published-margin').textContent = formatMargin(published.margin);
    document.getElementById('method-likelihood-margin').textContent = formatMargin(likelihood.margin);
    document.getElementById('method-election-margin').textContent = formatMargin(model.election_day?.mean);
    document.getElementById('method-probability-example').textContent = `${Math.round(houseData.summary.prob_dem_majority * 100)}% Democratic-majority probability`;

}

function createNationalTrendChart() {
    const canvas = document.getElementById('national-trend-chart');
    const history = (houseData.polling?.generic_ballot || [])
        .map(row => ({x: dateValue(row.date), y: row.margin}))
        .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
        .sort((a, b) => a.x - b.x);
    const likelihood = houseData.polling?.national_likelihood;
    if (!history.length) {
        componentMessage('national-trend-chart', 'Archived polling history is unavailable.');
        return;
    }
    if (!chartAvailable('national-trend-chart')) return;
    if (nationalTrendChart) nationalTrendChart.destroy();
    const finalInput = likelihood ? [{x: dateValue(likelihood.date), y: likelihood.margin}] : [];
    nationalTrendChart = new Chart(canvas, {
        type: 'line',
        data: {datasets: [
            {label: 'Silver published LV average', data: history, borderColor: '#60a5fa', backgroundColor: '#60a5fa', borderWidth: 2, pointRadius: 2, tension: 0, fill: false},
            {type: 'scatter', label: 'Final House polling input', data: finalInput, backgroundColor: '#fbbf24', borderColor: '#fbbf24', pointRadius: 6, pointStyle: 'rectRot'},
        ]},
        options: {
            responsive: true, maintainAspectRatio: false, parsing: false,
            scales: {
                x: dateAxis([...history, ...finalInput]),
                y: {title: {display: true, text: 'Democratic minus Republican margin', color: '#aaa'}, ticks: {color: '#aaa', callback: value => formatMargin(value)}, grid: {color: '#2a2a2a'}},
            },
            plugins: {
                legend: {labels: {color: '#aaa'}},
                tooltip: {callbacks: {title: items => chartDate(items[0].parsed.x), label: context => `${context.dataset.label}: ${formatMargin(context.parsed.y)}`}},
            },
        },
    });
}

// Update Senate statistics
function updateSenateStats() {
    const { summary, metadata, categories } = senateData;

    document.getElementById('senate-update-time').textContent = formatDate(new Date(metadata.updated_at));
    document.getElementById('senate-dem-prob').textContent = `${Math.round(summary.prob_dem_control * 100)}%`;
    document.getElementById('senate-median-seats').textContent = summary.median_dem_seats;
    document.getElementById('senate-confidence-interval').textContent = `${summary.ci_90_low}-${summary.ci_90_high}`;
    document.getElementById('senate-seats-up').textContent = summary.seats_up;
    document.getElementById('senate-dem-defending').textContent = summary.dem_defending;
    document.getElementById('senate-rep-defending').textContent = summary.rep_defending;

    updateCategoriesBar('senate-categories-bar', categories, 100, true);
}

// Update categories bar
function updateCategoriesBar(elementId, categories, total, isSenate = false) {
    const bar = document.getElementById(elementId);
    if (!bar) return;

    if (isSenate && senateData) {
        // For Senate, show all 100 seats with guaranteed seats at the ends
        const demNotUp = senateData.summary.dem_not_up;
        const repNotUp = senateData.summary.rep_not_up;

        const segments = [
            { class: 'guaranteed-d', count: demNotUp, label: 'Not Up (D)' },
            { class: 'safe-d', count: categories.safe_d || 0 },
            { class: 'likely-d', count: categories.likely_d || 0 },
            { class: 'lean-d', count: categories.lean_d || 0 },
            { class: 'tossup', count: categories.toss_up || 0 },
            { class: 'lean-r', count: categories.lean_r || 0 },
            { class: 'likely-r', count: categories.likely_r || 0 },
            { class: 'safe-r', count: categories.safe_r || 0 },
            { class: 'guaranteed-r', count: repNotUp, label: 'Not Up (R)' },
        ];

        bar.innerHTML = segments.map(seg => {
            const pct = (seg.count / 100 * 100).toFixed(1);
            const title = seg.label || seg.count;
            return `<div class="cat-segment ${seg.class}" style="flex-basis: ${pct}%"
                         title="${title}">${seg.count > 4 ? seg.count : ''}</div>`;
        }).join('');
    } else {
        // House - original behavior
        const segments = [
            { class: 'safe-d', count: categories.safe_d || categories.dem?.safe || 0 },
            { class: 'likely-d', count: categories.likely_d || categories.dem?.likely || 0 },
            { class: 'lean-d', count: categories.lean_d || categories.dem?.lean || 0 },
            { class: 'tossup', count: categories.toss_up || 0 },
            { class: 'lean-r', count: categories.lean_r || categories.rep?.lean || 0 },
            { class: 'likely-r', count: categories.likely_r || categories.rep?.likely || 0 },
            { class: 'safe-r', count: categories.safe_r || categories.rep?.safe || 0 },
        ];

        bar.innerHTML = segments.map(seg => {
            const pct = (seg.count / total * 100).toFixed(1);
            return `<div class="cat-segment ${seg.class}" style="flex-basis: ${pct}%"
                         title="${seg.count}">${seg.count > (total / 20) ? seg.count : ''}</div>`;
        }).join('');
    }
}

// Initialize Leaflet map
function initializeMap() {
    // Create map
    map = L.map('map-container', {
        center: [39.8, -98.5],
        zoom: 4,
        minZoom: 3,
        maxZoom: 10,
    });

    // Add the API-key-free OpenStreetMap basemap
    L.tileLayer(BASEMAP.url, {
        attribution: BASEMAP.attribution,
        maxZoom: 19
    }).addTo(map);

    // Initialize with House map (districts)
    updateMapLayer();
}

// Update map layer based on current chamber
function updateMapLayer() {
    // Remove existing layer
    if (geoLayer) {
        map.removeLayer(geoLayer);
    }

    if (currentChamber === 'house') {
        renderHouseMap();
    } else {
        renderSenateMap();
    }
}

// Render House map (congressional districts)
function renderHouseMap() {
    if (!districtGeoJSON) return;

    // Create district lookup
    const districtLookup = {};
    houseData.districts.forEach(d => {
        districtLookup[d.id] = d;
    });

    // Style function
    function getStyle(feature) {
        const districtId = feature.properties.district_id;
        const district = districtLookup[districtId];

        if (!district) {
            return {
                fillColor: '#333',
                weight: 0.5,
                opacity: 0.5,
                color: '#555',
                fillOpacity: 0.3
            };
        }

        return {
            fillColor: COLORS[district.category] || '#888',
            weight: 1,
            opacity: 0.7,
            color: '#333',
            fillOpacity: 0.7
        };
    }

    // Highlight on hover
    function highlightFeature(e) {
        const layer = e.target;
        layer.setStyle({
            weight: 2,
            color: '#fff',
            fillOpacity: 0.9
        });
        layer.bringToFront();
    }

    function resetHighlight(e) {
        geoLayer.resetStyle(e.target);
    }

    // Click handler
    function onFeatureClick(e) {
        const districtId = e.target.feature.properties.district_id;
        showDistrictModal(districtId);
    }

    // Add GeoJSON layer
    geoLayer = L.geoJSON(districtGeoJSON, {
        style: getStyle,
        onEachFeature: (feature, layer) => {
            const districtId = feature.properties.district_id;
            const district = districtLookup[districtId];

            if (district) {
                layer.bindTooltip(`
                    <div class="district-popup">
                        <h4>${district.id}</h4>
                        <p class="prob">${Math.round(district.prob_dem * 100)}% Dem</p>
                        <p>${district.incumbent.name} (${district.incumbent.party})</p>
                    </div>
                `, { sticky: true });
            }

            layer.on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: onFeatureClick
            });
        }
    }).addTo(map);
}

// Render Senate map (states)
function renderSenateMap() {
    if (!statesGeoJSON || !senateData) return;

    // Create race lookup by state
    const raceLookup = {};
    senateData.races.forEach(r => {
        raceLookup[r.state] = r;
    });

    // Style function for states
    function getStateStyle(feature) {
        const fipsCode = feature.id;
        const stateAbbr = FIPS_TO_STATE[fipsCode];

        if (!stateAbbr) {
            return {
                fillColor: '#333',
                weight: 0.5,
                opacity: 0.5,
                color: '#555',
                fillOpacity: 0.3
            };
        }

        const race = raceLookup[stateAbbr];

        // State has a race up in 2026
        if (race) {
            return {
                fillColor: COLORS[race.category] || '#888',
                weight: 1.5,
                opacity: 0.8,
                color: '#fff',
                fillOpacity: 0.75
            };
        }

        // State not up in 2026 - show as dark grey
        return {
            fillColor: '#444',
            weight: 1,
            opacity: 0.5,
            color: '#333',
            fillOpacity: 0.4
        };
    }

    // Highlight on hover
    function highlightFeature(e) {
        const layer = e.target;
        layer.setStyle({
            weight: 3,
            color: '#fff',
            fillOpacity: 0.9
        });
        layer.bringToFront();
    }

    function resetHighlight(e) {
        geoLayer.resetStyle(e.target);
    }

    // Click handler
    function onStateClick(e) {
        const fipsCode = e.target.feature.id;
        const stateAbbr = FIPS_TO_STATE[fipsCode];
        if (stateAbbr && raceLookup[stateAbbr]) {
            showDistrictModal(stateAbbr);
        }
    }

    // Add GeoJSON layer
    geoLayer = L.geoJSON(statesGeoJSON, {
        style: getStateStyle,
        onEachFeature: (feature, layer) => {
            const fipsCode = feature.id;
            const stateAbbr = FIPS_TO_STATE[fipsCode];
            const stateName = feature.properties.name;
            const race = raceLookup[stateAbbr];

            if (race) {
                // State has a 2026 race
                layer.bindTooltip(`
                    <div class="district-popup">
                        <h4>${stateName} (${stateAbbr})</h4>
                        <p class="prob">${Math.round(race.prob_dem * 100)}% Dem</p>
                        <p>${race.incumbent} (${race.incumbent_party})</p>
                        <p class="up-2026">Up in 2026</p>
                    </div>
                `, { sticky: true });
            } else if (stateAbbr) {
                // State not up in 2026
                layer.bindTooltip(`
                    <div class="district-popup">
                        <h4>${stateName} (${stateAbbr})</h4>
                        <p class="not-up">Not up in 2026</p>
                    </div>
                `, { sticky: true });
            }

            layer.on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: onStateClick
            });
        }
    }).addTo(map);
}

// Create seat distribution chart
function createSeatChart() {
    const data = currentChamber === 'house' ? houseData : senateData;
    if (!data) return;
    const {seat_distribution, summary} = data;
    const key = currentChamber === 'house' ? 'prob_dem_majority' : 'prob_dem_control';
    const probLabel = currentChamber === 'house' ? 'Majority' : 'Control';
    document.getElementById('chart-stats').innerHTML = `
        <div><span>Dem ${probLabel}:</span> <strong>${Math.round(summary[key] * 100)}%</strong></div>
        <div><span>Median:</span> <strong>${summary.median_dem_seats} seats</strong></div>
        <div><span>90% CI:</span> <strong>${summary.ci_90_low}-${summary.ci_90_high}</strong></div>`;
    if (!chartAvailable('seat-chart')) return;
    if (!seat_distribution?.dem_seats?.length) {
        componentMessage('seat-chart', 'Archived seat distribution is unavailable.');
        return;
    }
    if (seatChart) seatChart.destroy();
    const threshold = currentChamber === 'house' ? 218 : 51;
    seatChart = new Chart(document.getElementById('seat-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: seat_distribution.dem_seats,
            datasets: [{
                data: seat_distribution.probabilities.map(p => p * 100),
                backgroundColor: seat_distribution.dem_seats.map(seats => seats >= threshold ? 'rgba(0, 21, 188, 0.8)' : 'rgba(188, 0, 0, 0.8)'),
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {legend: {display: false}, tooltip: {callbacks: {label: context => `${context.parsed.y.toFixed(2)}% probability`}}},
            scales: {
                x: {title: {display: true, text: 'Democratic seats', color: '#aaa'}, ticks: {color: '#aaa', maxTicksLimit: 20}, grid: {display: false}},
                y: {title: {display: true, text: 'Probability (%)', color: '#aaa'}, ticks: {color: '#aaa'}, grid: {color: '#2a2a2a'}},
            },
        },
    });
}

function timelineRows() {
    const key = currentTimelineChamber === 'house' ? 'prob_dem_majority' : 'prob_dem_control';
    return [...(currentTimelineChamber === 'house' ? houseTimeline || [] : senateTimeline || [])]
        .filter(row => Number.isFinite(dateValue(row.date)) && Number.isFinite(row[key]))
        .sort((a, b) => dateValue(a.date) - dateValue(b.date));
}

function createTimelineChart() {
    const data = timelineRows();
    updateTimelineStats(data);
    if (timelineChart) {
        timelineChart.destroy();
        timelineChart = null;
    }
    if (!data.length) {
        componentMessage('timeline-chart', 'Archived timeline data is unavailable for this chamber.');
        return;
    }
    if (!chartAvailable('timeline-chart')) return;
    const key = currentTimelineChamber === 'house' ? 'prob_dem_majority' : 'prob_dem_control';
    const points = data.map(row => ({x: dateValue(row.date), y: row[key] * 100}));
    timelineChart = new Chart(document.getElementById('timeline-chart').getContext('2d'), {
        type: 'line',
        data: {datasets: [{
            label: 'Democratic control probability', data: points,
            borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.15)',
            borderWidth: 2, fill: true, tension: 0, pointRadius: 2, pointHoverRadius: 6,
        }]},
        options: {
            responsive: true, maintainAspectRatio: false, parsing: false,
            plugins: {
                legend: {display: false},
                tooltip: {callbacks: {
                    title: items => chartDate(items[0].parsed.x),
                    label: context => `${context.parsed.y.toFixed(2)}% probability`,
                    afterLabel: context => `Median: ${data[context.dataIndex].median_dem_seats} seats`,
                }},
            },
            scales: {
                x: dateAxis(points),
                y: {title: {display: true, text: 'Democratic control probability (%)', color: '#aaa'}, ticks: {color: '#aaa'}, grid: {color: '#2a2a2a'}, min: 0, max: 100},
            },
        },
    });
}

function updateTimelineStats(data) {
    const target = document.getElementById('timeline-stats');
    if (data.length < 2) {
        target.textContent = 'Two saved forecasts are needed to compare changes.';
        return;
    }
    const key = currentTimelineChamber === 'house' ? 'prob_dem_majority' : 'prob_dem_control';
    const previous = data[data.length - 2];
    const latest = data[data.length - 1];
    const change = (latest[key] - previous[key]) * 100;
    const chamber = currentTimelineChamber === 'house' ? 'House' : 'Senate';
    target.textContent = `${chamber} Democratic control probability: ${(previous[key] * 100).toFixed(2)}% on ${chartDate(dateValue(previous.date))}, 2026 → ${(latest[key] * 100).toFixed(2)}% on ${chartDate(dateValue(latest.date))}, 2026 (${change >= 0 ? '+' : ''}${change.toFixed(2)} percentage points).`;
}

// Populate state filter
function populateStateFilter() {
    const races = currentChamber === 'house' ? houseData.districts : (senateData?.races || []);
    const states = [...new Set(races.map(d => d.state))].sort();
    const select = document.getElementById('state-filter');

    select.innerHTML = '<option value="">All States</option>';
    states.forEach(state => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = state;
        select.appendChild(option);
    });
}

// Render table
function renderTable(filter = {}) {
    const isHouse = currentChamber === 'house';
    let races = isHouse ? [...houseData.districts] : [...(senateData?.races || [])];

    // Update title
    document.getElementById('table-title').textContent = isHouse ? 'All 435 Districts' : 'All Senate Races';
    document.getElementById('total-count').textContent = races.length;

    // Apply filters
    if (filter.state) {
        races = races.filter(d => d.state === filter.state);
    }
    if (filter.category) {
        races = races.filter(d => d.category === filter.category);
    }
    if (filter.search) {
        const search = filter.search.toLowerCase();
        races = races.filter(d =>
            (d.id || d.state).toLowerCase().includes(search) ||
            (d.incumbent?.name || d.incumbent || '').toLowerCase().includes(search)
        );
    }

    // Sort
    races.sort((a, b) => {
        let aVal = a[sortColumn];
        let bVal = b[sortColumn];

        if (sortColumn === 'incumbent') {
            aVal = a.incumbent?.name || a.incumbent || '';
            bVal = b.incumbent?.name || b.incumbent || '';
        }
        if (sortColumn === 'id') {
            aVal = a.id || a.state;
            bVal = b.id || b.state;
        }

        if (typeof aVal === 'string') {
            return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        }
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });

    // Render
    const tbody = document.getElementById('district-tbody');
    tbody.innerHTML = races.map(d => {
        const id = d.id || d.state;
        const incName = d.incumbent?.name || d.incumbent || 'Open';
        const incParty = d.incumbent?.party || d.incumbent_party || '';

        return `
            <tr data-district="${id}">
                <td class="district-id">${id}</td>
                <td>
                    ${incParty ? `<span class="incumbent-party ${incParty}">${incParty}</span>` : ''}
                    ${incName}
                </td>
                <td class="pvi">${formatMargin(d.prior_margin)}</td>
                <td class="pvi">${formatMargin(d.posterior_margin)}</td>
                <td class="prob">${Math.round(d.prob_dem * 100)}%</td>
                <td class="pvi">${d.polls_used ? 'Silver' : '—'}</td>
                <td><span class="rating ${d.category}">${formatCategory(d.category)}</span></td>
            </tr>
        `;
    }).join('');

    document.getElementById('showing-count').textContent = races.length;
}

// Setup event listeners
function setupEventListeners() {
    // Chamber toggle
    document.querySelectorAll('.chamber-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            selectChamber(btn.dataset.chamber);
        });
    });

    // Timeline toggle
    document.querySelectorAll('.timeline-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.timeline-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimelineChamber = btn.dataset.timeline;
            renderOptional('timeline-chart', createTimelineChart);
        });
    });

    // Filters
    document.getElementById('state-filter').addEventListener('change', applyFilters);
    document.getElementById('category-filter').addEventListener('change', applyFilters);
    document.getElementById('search-filter').addEventListener('input', debounce(applyFilters, 300));

    // Table sorting
    document.querySelectorAll('.district-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.sort;
            if (sortColumn === column) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = column;
                sortDirection = column === 'prob_dem' ? 'desc' : 'asc';
            }
            applyFilters();
        });
    });

    // Row click
    document.getElementById('district-tbody').addEventListener('click', (e) => {
        const row = e.target.closest('tr');
        if (row) showDistrictModal(row.dataset.district);
    });

    // Modal
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('district-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

function applyFilters() {
    renderTable({
        state: document.getElementById('state-filter').value,
        category: document.getElementById('category-filter').value,
        search: document.getElementById('search-filter').value,
    });
}

function showDistrictModal(id) {
    const isHouse = currentChamber === 'house';
    const races = isHouse ? houseData.districts : (senateData?.races || []);
    const race = races.find(d => (d.id || d.state) === id);
    if (!race) return;

    const modal = document.getElementById('district-modal');
    const body = document.getElementById('modal-body');

    const incName = race.incumbent?.name || race.incumbent || 'Open Seat';
    const incParty = race.incumbent?.party || race.incumbent_party || '';

    body.innerHTML = `
        <div class="modal-header">
            <h3>${race.id || race.state}</h3>
            <p>${incName}${incParty ? ` (${incParty})` : ''}</p>
        </div>
        <div class="modal-stats">
            <div class="modal-stat">
                <div class="modal-stat-value">${Math.round(race.prob_dem * 100)}%</div>
                <div class="modal-stat-label">Democratic Win Probability</div>
            </div>
            <div class="modal-stat">
                <div class="modal-stat-value">${formatPVI(race.pvi)}</div>
                <div class="modal-stat-label">Partisan Lean (PVI)</div>
            </div>
            ${race.mean_vote_share ? `
            <div class="modal-stat">
                <div class="modal-stat-value">${race.mean_vote_share.toFixed(1)}%</div>
                <div class="modal-stat-label">Expected Dem Vote Share</div>
            </div>
            <div class="modal-stat">
                <div class="modal-stat-value">${race.ci_90_low?.toFixed(0)}-${race.ci_90_high?.toFixed(0)}%</div>
                <div class="modal-stat-label">90% Credible Interval</div>
            </div>
            ` : ''}
            <div class="modal-stat"><div class="modal-stat-value">${formatMargin(race.prior_margin)}</div><div class="modal-stat-label">Fundamentals Prior</div></div>
                <div class="modal-stat"><div class="modal-stat-value">${formatMargin(race.posterior_margin)}</div><div class="modal-stat-label">Updated posterior (${race.polls_used ? 'Silver average' : 'fundamentals only'})</div></div>
        </div>
        <div style="text-align: center; margin-top: 16px;">
            <span class="rating ${race.category}" style="font-size: 1rem; padding: 8px 16px;">
                ${formatCategory(race.category)}
            </span>
        </div>
    `;

    modal.classList.add('active');
}

function closeModal() {
    document.getElementById('district-modal').classList.remove('active');
}

// Update map legend based on chamber
function updateMapLegend() {
    const legendContainer = document.querySelector('.map-legend');
    const mapTitle = document.getElementById('map-title');
    const mapDesc = document.getElementById('map-desc');

    if (currentChamber === 'house') {
        if (mapTitle) mapTitle.textContent = 'Congressional District Map';
        if (mapDesc) mapDesc.textContent = 'Click any district for detailed forecast. Color indicates race rating.';
        if (legendContainer) {
            legendContainer.innerHTML = `
                <span class="legend-item safe-d">Safe D</span>
                <span class="legend-item likely-d">Likely D</span>
                <span class="legend-item lean-d">Lean D</span>
                <span class="legend-item tossup">Toss-up</span>
                <span class="legend-item lean-r">Lean R</span>
                <span class="legend-item likely-r">Likely R</span>
                <span class="legend-item safe-r">Safe R</span>
            `;
        }
    } else {
        if (mapTitle) mapTitle.textContent = 'Senate Race Map';
        if (mapDesc) mapDesc.textContent = 'Click any state with a 2026 race for details. Grey states are not up for election.';
        if (legendContainer) {
            legendContainer.innerHTML = `
                <span class="legend-item safe-d">Safe D</span>
                <span class="legend-item likely-d">Likely D</span>
                <span class="legend-item lean-d">Lean D</span>
                <span class="legend-item tossup">Toss-up</span>
                <span class="legend-item lean-r">Lean R</span>
                <span class="legend-item likely-r">Likely R</span>
                <span class="legend-item safe-r">Safe R</span>
                <span class="legend-item not-up">Not Up</span>
            `;
        }
    }
}

// Utility functions
function formatDate(date) {
    return date.toLocaleDateString('en-US', {
        month: 'long', day: 'numeric', year: 'numeric',
        hour: 'numeric', minute: '2-digit', timeZoneName: 'short', timeZone: 'America/Los_Angeles'
    });
}

function formatPVI(pvi) {
    if (pvi === 0) return 'EVEN';
    return pvi > 0 ? `D+${pvi}` : `R+${Math.abs(pvi)}`;
}

function formatCategory(category) {
    return {
        safe_d: 'Safe D', likely_d: 'Likely D', lean_d: 'Lean D',
        toss_up: 'Toss-up',
        lean_r: 'Lean R', likely_r: 'Likely R', safe_r: 'Safe R',
    }[category] || category;
}

function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

// Mobile menu toggle
function setupMobileMenu() {
    const menuBtn = document.getElementById('mobile-menu-btn');
    const mobileNav = document.getElementById('mobile-nav');

    if (menuBtn && mobileNav) {
        menuBtn.addEventListener('click', () => {
            menuBtn.classList.toggle('active');
            mobileNav.classList.toggle('active');
        });

        // Close menu when clicking a link
        mobileNav.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                menuBtn.classList.remove('active');
                mobileNav.classList.remove('active');
            });
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setupMobileMenu();
});
