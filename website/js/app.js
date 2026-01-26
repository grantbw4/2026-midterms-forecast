/**
 * 2026 Midterm Elections Forecast - Frontend Application
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
let sortColumn = 'prob_dem';
let sortDirection = 'desc';

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

// Load all data
async function loadData() {
    const paths = ['./forecast.json', '../outputs/forecast.json'];
    const senatePaths = ['./senate_forecast.json', '../outputs/senate_forecast.json'];
    const geoPaths = ['./data/districts.geojson', '../website/data/districts.geojson'];
    const statePaths = ['./data/states.geojson', '../website/data/states.geojson'];

    // Load House forecast
    for (const path of paths) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                houseData = await response.json();
                break;
            }
        } catch (e) { }
    }

    // Load Senate forecast
    for (const path of senatePaths) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                senateData = await response.json();
                break;
            }
        } catch (e) { }
    }

    // Load district GeoJSON
    for (const path of geoPaths) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                districtGeoJSON = await response.json();
                break;
            }
        } catch (e) { }
    }

    // Load states GeoJSON
    for (const path of statePaths) {
        try {
            const response = await fetch(path);
            if (response.ok) {
                statesGeoJSON = await response.json();
                break;
            }
        } catch (e) { }
    }

    if (!houseData) {
        document.body.innerHTML = `
            <div style="padding: 48px; text-align: center; color: #888;">
                <h2>Unable to load forecast data</h2>
                <p>Make sure forecast.json exists.</p>
            </div>
        `;
        return;
    }

    initializePage();
}

// Initialize all page components
function initializePage() {
    updateHouseStats();
    if (senateData) updateSenateStats();
    if (districtGeoJSON) initializeMap();
    createSeatChart();
    populateStateFilter();
    renderTable();
    setupEventListeners();
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

    // Environment
    const envValue = summary.national_environment;
    document.getElementById('national-env').textContent = envValue >= 0 ? `D+${envValue.toFixed(1)}` : `R+${Math.abs(envValue).toFixed(1)}`;
    const gbValue = summary.generic_ballot_margin;
    document.getElementById('generic-ballot').textContent = gbValue >= 0 ? `D+${gbValue.toFixed(1)}` : `R+${Math.abs(gbValue).toFixed(1)}`;
    document.getElementById('approval').textContent = `${summary.approval_rating.toFixed(0)}%`;
    document.getElementById('days-until').textContent = metadata.days_until_election;

    // Categories bar
    updateCategoriesBar('house-categories-bar', categories, 435);
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
        const seatsUp = senateData.summary.seats_up || 33;
        const seatsNotUp = 100 - seatsUp;

        // Seats not up: 100 - seatsUp, split roughly
        // Current Senate is roughly 47D-53R, so seats not up follow that pattern
        const demNotUp = Math.round(seatsNotUp * 0.47);
        const repNotUp = seatsNotUp - demNotUp;

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

    // Add tile layer (dark theme)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap, &copy; CARTO',
        subdomains: 'abcd',
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

    const { seat_distribution, summary } = data;
    const ctx = document.getElementById('seat-chart').getContext('2d');

    // Destroy existing chart
    if (seatChart) seatChart.destroy();

    const threshold = currentChamber === 'house' ? 218 : 50;
    const backgroundColors = seat_distribution.dem_seats.map(seats =>
        seats >= threshold ? 'rgba(0, 21, 188, 0.8)' : 'rgba(188, 0, 0, 0.8)'
    );

    seatChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: seat_distribution.dem_seats,
            datasets: [{
                data: seat_distribution.probabilities.map(p => p * 100),
                backgroundColor: backgroundColors,
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y.toFixed(2)}% probability`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Democratic Seats', color: '#888' },
                    ticks: { color: '#888', maxTicksLimit: 20 },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Probability (%)', color: '#888' },
                    ticks: { color: '#888' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });

    // Update stats
    const key = currentChamber === 'house' ? 'prob_dem_majority' : 'prob_dem_control';
    const probLabel = currentChamber === 'house' ? 'Majority' : 'Control';
    document.getElementById('chart-stats').innerHTML = `
        <div><span>Dem ${probLabel}:</span> <strong>${Math.round(summary[key] * 100)}%</strong></div>
        <div><span>Median:</span> <strong>${summary.median_dem_seats} seats</strong></div>
        <div><span>90% CI:</span> <strong>${summary.ci_90_low}-${summary.ci_90_high}</strong></div>
    `;
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
                <td class="pvi">${formatPVI(d.pvi)}</td>
                <td class="prob">${Math.round(d.prob_dem * 100)}%</td>
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
            document.querySelectorAll('.chamber-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentChamber = btn.dataset.chamber;

            document.querySelectorAll('.chamber-section').forEach(s => s.classList.remove('active'));
            document.querySelector(`.chamber-section[data-chamber="${currentChamber}"]`)?.classList.add('active');

            // Update map to show districts or states
            if (map) updateMapLayer();

            // Update map legend
            updateMapLegend();

            createSeatChart();
            populateStateFilter();
            renderTable();
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
                <div class="modal-stat-label">90% Confidence Interval</div>
            </div>
            ` : ''}
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
        hour: 'numeric', minute: '2-digit', timeZoneName: 'short'
    });
}

function formatPVI(pvi) {
    if (pvi === 0) return 'EVEN';
    return pvi > 0 ? `R+${pvi}` : `D+${Math.abs(pvi)}`;
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
