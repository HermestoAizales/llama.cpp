<script lang="ts">
	export interface TokenLogprobData {
		token: string;
		logprob: number;
		top_logprobs: Array<{ token: string; logprob: number }>;
	}

	interface Props {
		token: TokenLogprobData | null;
		index: number;
		anchorEl: HTMLElement | null;
	}

	let { token, index, anchorEl }: Props = $props();

	let popoverStyle = $state('position: fixed; display: none;');

	function computeEntropy(data: TokenLogprobData): number {
		let h = 0;
		const items =
			data.top_logprobs.length > 0
				? data.top_logprobs
				: [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}

	function formatToken(t: string): string {
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/ /g, '\u00B7');
	}

	function getProbBadge(prob: number): string {
		if (prob >= 0.9) return 'text-[hsl(var(--chart-2))]';
		if (prob >= 0.5) return 'text-[hsl(45_93%_47%)]';
		return 'text-[hsl(var(--destructive))]';
	}

	function getBarBg(prob: number): string {
		if (prob >= 0.9) return 'bg-[hsl(var(--chart-2))]';
		if (prob >= 0.5) return 'bg-[hsl(45_93%_47%)]';
		return 'bg-[hsl(var(--destructive))]';
	}

	$effect(() => {
		if (!token || !token || !anchorEl) {
			popoverStyle = 'position: fixed; display: none;';
			return;
		}
		const r = anchorEl.getBoundingClientRect();
		if (r.width === 0) {
			popoverStyle = 'position: fixed; display: none;';
			return;
		}
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		const pw = 320;
		const ph = 240;

		let left = r.left + r.width / 2;
		let top = r.bottom + 6;

		if (left + pw / 2 > vw - 4) left = vw - pw / 2 - 4;
		if (left - pw / 2 < 4) left = pw / 2 + 4;
		if (top + ph > vh) top = r.top - ph - 6;
		if (top < 4) top = 4;

		popoverStyle = `position: fixed; left: ${left}px; top: ${top}px; transform: translateX(-50%); z-index: 9999; display: block;`;
	});
</script>

{#if token}
	<div
		class="pointer-events-none w-[320px] max-h-[240px] overflow-hidden rounded-xl border border-border/60 bg-popover/95 px-3 py-2.5 text-xs shadow-xl backdrop-blur-md"
		style={popoverStyle}
	>
		<div class="mb-1.5 flex items-center gap-1.5">
			<span class="w-5 shrink-0 text-right tabular-nums font-mono font-bold text-muted-foreground select-none">
				#{index}
			</span>
			<span class="max-w-[140px] truncate font-mono font-semibold" title="{token.token}">
				{formatToken(token.token)}
			</span>
			{#if token.logprob}
				{@const prob = Math.exp(token.logprob)}
				<span class="shrink-0 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold tabular-nums border {getProbBadge(prob)}">
					{(prob * 100).toFixed(1)}%
				</span>
			{/if}
			<span class="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground tabular-nums">
				logp: {token.logprob.toFixed(3)}
			</span>
		</div>

		{#if token.top_logprobs && token.top_logprobs.length > 0}
			{@const entropy = computeEntropy(token)}
			<div class="border-t border-border/30 pt-1.5">
				<div class="mb-1 flex items-center justify-between text-[10px] tabular-nums text-muted-foreground">
					<span>Top-{token.top_logprobs.length} alternatives</span>
					<span>H={entropy.toFixed(2)}</span>
				</div>
				{#each token.top_logprobs as alt, ai (alt.token + ai)}
					{@const altProb = Math.exp(alt.logprob)}
					<div class="flex items-center gap-1 py-0.5">
						<span class="w-3 shrink-0 text-right text-[9px] text-muted-foreground">{ai + 1}.</span>
						<span class="w-[90px] shrink-0 truncate font-mono text-[10px]" title="{alt.token}">
							{formatToken(alt.token)}
						</span>
						<div class="h-2.5 min-w-[16px] flex-1 overflow-hidden rounded-sm bg-muted/40">
							<div
								class="h-full rounded-sm {getBarBg(altProb)}"
								style={'width: ' + Math.min(altProb * 100, 100) + '%'}
							></div>
						</div>
						<span class="w-9 shrink-0 text-right text-[9px] tabular-nums text-muted-foreground">
							{(altProb * 100).toFixed(1)}%
						</span>
					</div>
				{/each}
			</div>
		{:else if token.logprob !== 0}
			{@const entropy = -Math.exp(token.logprob) * Math.log2(Math.exp(token.logprob))}
			<div class="border-t border-border/30 pt-1 text-[10px] text-muted-foreground">
				<span class="tabular-nums">H={Math.abs(entropy).toFixed(2)}</span>
			</div>
		{/if}
	</div>
{/if}
